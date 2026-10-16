"""Tests for tec.inference and tec.tflite_export."""

import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tec import configs  # noqa: E402
from tec import inference  # noqa: E402
from tec import tflite_export  # noqa: E402
from lingvo import compat as tf  # noqa: E402
from lingvo.core import test_utils  # noqa: E402


class _TinyTecConfig(configs.TecSingleInterfering):
  """Compact TEC configuration for fast unit test inference and TFLite."""

  def Task(self):
    p = super().Task()
    p.encoder_speech.conv_filter_shapes = [(3, 3, 1, 4), (3, 3, 4, 4)]
    p.encoder_speech.lstm_cell_size = 8
    p.encoder_speech.num_lstm_layers = 1
    p.encoder.emb.embedding_dim = 16
    p.encoder.filter_shapes = [(3, 1, 16, 16)]
    p.encoder.filter_strides = [(1, 1)]
    p.encoder.lstm_cell_size = 8
    p.decoder.source_dim = 16
    p.decoder.step.rnn_cell_dim = 8
    p.decoder.step.rnn_layers = 1
    p.decoder.step.target_pre_net.hidden_layer_dims = [8, 8]
    p.decoder.post_net.hidden_channels = 8
    p.decoder.post_net.num_layers = 2
    p.decoder.decode_max_output_frames = 4
    p.waveform_processor.num_griffin_lim_iters = 2
    return p


class _TinyAecConfig(configs.AecSingleInterfering):
  """Compact AEC configuration for unit test inference and TFLite."""

  def Task(self):
    p = super().Task()
    p.encoder_speech.conv_filter_shapes = [(3, 3, 1, 4), (3, 3, 4, 4)]
    p.encoder_speech.lstm_cell_size = 8
    p.encoder_speech.num_lstm_layers = 1
    p.encoder_interfering.conv_filter_shapes = [(3, 3, 1, 4), (3, 3, 4, 4)]
    p.encoder_interfering.lstm_cell_size = 8
    p.encoder_interfering.num_lstm_layers = 1
    p.decoder.source_dim = 16
    p.decoder.step.rnn_cell_dim = 8
    p.decoder.step.rnn_layers = 1
    p.decoder.step.target_pre_net.hidden_layer_dims = [8, 8]
    p.decoder.post_net.hidden_channels = 8
    p.decoder.post_net.num_layers = 2
    p.decoder.decode_max_output_frames = 4
    p.waveform_processor.num_griffin_lim_iters = 2
    return p


class InferenceAndTfliteTest(test_utils.TestCase):

  def testInferenceRunnerDeterministicAndTextSensitive(self):
    runner = inference.TecInferenceEngine(
        model_config_cls=_TinyTecConfig, decode_max_output_frames=4)
    try:
      wav = np.sin(np.linspace(0.0, 20.0, 2400, dtype=np.float32)) * 0.2
      out1 = runner.predict(
          mixed_waveforms=[wav], interfering_texts=['good morning'])
      out2 = runner.predict(
          mixed_waveforms=[wav], interfering_texts=['good morning'])
      out_rev = runner.predict(
          mixed_waveforms=[wav], interfering_texts=['gninrom doog'])
      self.assertEqual((1, 4, 128), out1['feature_preds'].shape)
      self.assertEqual(1, out1['predicted_waveforms'].shape[0])
      # Verify eval mode makes repeated inference calls 100% deterministic.
      np.testing.assert_allclose(
          out1['feature_preds'], out2['feature_preds'], atol=0.0, rtol=0.0)
      # Verify same-length different-content text alters predictions.
      diff = np.abs(out1['feature_preds'] - out_rev['feature_preds'])
      self.assertGreater(float(np.max(diff)), 1e-4)
    finally:
      runner.close()

  def testAecInferenceRequiresInterferingWaveform(self):
    runner = inference.TecInferenceRunner(
        model_config_cls=_TinyAecConfig, decode_max_output_frames=4)
    try:
      wav = np.zeros(2400, dtype=np.float32)
      with self.assertRaises(ValueError):
        runner.predict(mixed_waveforms=[wav])
      out = runner.predict(
          mixed_waveforms=[wav], interfering_waveforms=[wav])
      self.assertEqual((1, 4, 128), out['feature_preds'].shape)
    finally:
      runner.close()

  def testExportAndRunTfliteTecAndAec(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      tflite_path = os.path.join(tmpdir, 'tec_model.tflite')
      model_bytes = tflite_export.export_to_tflite(
          output_tflite_path=tflite_path,
          model_config_cls=_TinyTecConfig,
          num_frames=8,
          text_length=6,
          decode_steps=3)
      self.assertTrue(os.path.exists(tflite_path))
      self.assertGreater(len(model_bytes), 0)

      src_feat = np.zeros((1, 8, 128), dtype=np.float32)
      ids = np.ones((1, 6), dtype=np.int32) * 10
      out1 = tflite_export.run_tflite_inference(
          tflite_path, source_features=src_feat, interfering_ids=ids)
      out2 = tflite_export.run_tflite_inference(
          tflite_path, source_features=src_feat, interfering_ids=ids)
      self.assertEqual((1, 3, 128), out1['feature_preds'].shape)
      self.assertEqual((1, 3, 1), out1['eos_probs'].shape)
      np.testing.assert_allclose(
          out1['feature_preds'], out2['feature_preds'], atol=0.0, rtol=0.0)

      # Verify high-level inference.run_tflite_inference with 2D mel + text.
      mel_2d = np.zeros((5, 128), dtype=np.float32)
      out_hl = inference.run_tflite_inference(
          tflite_path=tflite_path,
          mixed_mel=mel_2d,
          interfering_text='hello world')
      self.assertEqual((1, 3, 128), out_hl['feature_preds'].shape)
      self.assertIn('predicted_mel', out_hl)

      # Verify AecModel TFLite export exposes distinct interfering_features.
      aec_tflite_path = os.path.join(tmpdir, 'aec_model.tflite')
      aec_bytes = tflite_export.export_to_tflite(
          output_tflite_path=aec_tflite_path,
          model_config_cls=_TinyAecConfig,
          num_frames=8,
          decode_steps=3)
      self.assertGreater(len(aec_bytes), 0)
      int_feat_a = np.zeros((1, 8, 128), dtype=np.float32)
      int_feat_b = np.linspace(
          0.0, 2.0, 8 * 128, dtype=np.float32).reshape((1, 8, 128))
      aec_out_a = tflite_export.run_tflite_inference(
          aec_tflite_path,
          source_features=src_feat,
          interfering_features=int_feat_a)
      aec_out_b = tflite_export.run_tflite_inference(
          aec_tflite_path,
          source_features=src_feat,
          interfering_features=int_feat_b)
      self.assertEqual((1, 3, 128), aec_out_a['feature_preds'].shape)
      self.assertGreater(
          float(np.max(np.abs(
              aec_out_a['feature_preds'] - aec_out_b['feature_preds']))),
          1e-5)


if __name__ == '__main__':
  tf.test.main()
