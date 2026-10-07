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


class InferenceAndTfliteTest(test_utils.TestCase):

  def testInferenceRunner(self):
    runner = inference.TecInferenceRunner(
        model_config_cls=_TinyTecConfig, decode_max_output_frames=4)
    try:
      wav = np.zeros(2400, dtype=np.float32)
      outputs = runner.predict(
          mixed_waveforms=[wav], interfering_texts=['good morning'])
      self.assertEqual((1, 4, 128), outputs['feature_preds'].shape)
      self.assertEqual(1, outputs['predicted_waveforms'].shape[0])
    finally:
      runner.close()

  def testExportAndRunTflite(self):
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
      ids = np.ones((1, 6), dtype=np.int32)
      out = tflite_export.run_tflite_inference(
          tflite_path, source_features=src_feat, interfering_ids=ids)
      self.assertEqual((1, 3, 128), out['feature_preds'].shape)
      self.assertEqual((1, 3, 1), out['eos_probs'].shape)


if __name__ == '__main__':
  tf.test.main()
