"""Tests for tec.tec_model, tec.aec_model, and tec.configs."""

import os
import sys
import tempfile
from lingvo import compat as tf
from lingvo.core import test_utils
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tec import aec_model  # noqa: E402
from tec import configs  # noqa: E402
from tec import data_prep  # noqa: E402
from tec import input_generator  # noqa: E402


def _shrink_task_params(task_p):
  """Reduces model dimensions for fast unit test execution."""
  task_p.encoder_speech.conv_filter_shapes = [(3, 3, 1, 4), (3, 3, 4, 4)]
  task_p.encoder_speech.lstm_cell_size = 8
  task_p.encoder_speech.num_lstm_layers = 1
  if task_p.encoder is not None:
    task_p.encoder.emb.embedding_dim = 16
    task_p.encoder.filter_shapes = [(3, 1, 16, 16)]
    task_p.encoder.filter_strides = [(1, 1)]
    task_p.encoder.lstm_cell_size = 8
  if hasattr(task_p, 'encoder_interfering') and task_p.encoder_interfering:
    task_p.encoder_interfering.conv_filter_shapes = [
        (3, 3, 1, 4),
        (3, 3, 4, 4),
    ]
    task_p.encoder_interfering.lstm_cell_size = 8
    task_p.encoder_interfering.num_lstm_layers = 1
  task_p.decoder.source_dim = 16
  task_p.decoder.step.rnn_cell_dim = 8
  task_p.decoder.step.rnn_layers = 1
  task_p.decoder.step.target_pre_net.hidden_layer_dims = [8, 8]
  task_p.decoder.post_net.hidden_channels = 8
  task_p.decoder.post_net.num_layers = 2
  task_p.decoder.decode_max_output_frames = 4
  if task_p.waveform_processor:
    task_p.waveform_processor.num_griffin_lim_iters = 2
  return task_p


class TecAndAecModelTest(test_utils.TestCase):

  def _run_model_check(self, cfg_cls):
    with self.session(graph=tf.Graph(), use_gpu=False) as sess:
      cfg = cfg_cls()
      inp_p = cfg.Train()
      inp_p.batch_size = 2
      inp_p.use_synthetic_data = True
      inp_p.synthetic_num_samples = 2400
      inp_p.synthetic_text_len = 6
      inp = inp_p.Instantiate()

      task_p = _shrink_task_params(cfg.Task())
      task = task_p.Instantiate()

      batch = inp.GetPreprocessedInputBatch()
      preds = task.ComputePredictions(task.theta, batch)
      metrics, _ = task.ComputeLoss(task.theta, preds, batch)
      dec_out = task.Decode(batch)

      sess.run(tf.global_variables_initializer())
      loss_val, dec_dict = sess.run([metrics['loss'][0], dec_out])
      self.assertGreater(loss_val, 0.0)
      self.assertEqual((2, 4, 128), dec_dict['feature_preds'].shape)
      self.assertIn('predicted_waveforms', dec_dict)

      dec_metrics = task.CreateDecoderMetrics()
      post = task.PostProcessDecodeOut(dec_dict, dec_metrics)
      self.assertEqual(2, len(post))

  def testTecSingleInterferingModel(self):
    self._run_model_check(configs.TecSingleInterfering)

  def testNoSideInputModel(self):
    self._run_model_check(configs.NoSideInputSingleInterfering)

  def testAecSingleInterferingModel(self):
    self._run_model_check(configs.AecSingleInterfering)

  def testTfRecordInputGenerator(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      tfrecord_path = os.path.join(tmpdir, 'test.tfrecord')
      clean = [
          data_prep.UtteranceRecord(
              utt_id='c1',
              waveform=np.sin(
                  np.linspace(0, 10, 2400, dtype=np.float32)) * 0.2,
              transcript='turn off the lights')
      ]
      interfering = [
          data_prep.UtteranceRecord(
              utt_id='i1',
              waveform=np.cos(
                  np.linspace(0, 20, 3000, dtype=np.float32)) * 0.2,
              transcript='playing classical music')
      ]
      written = data_prep.prepare_tfrecord_dataset(
          clean, interfering, tfrecord_path)
      self.assertEqual(1, written)

      with self.session(graph=tf.Graph(), use_gpu=False) as sess:
        p = input_generator.TecInputGenerator.Params()
        p.file_pattern = tfrecord_path
        p.batch_size = 1
        p.shuffle = False
        inp = p.Instantiate()
        batch = inp.GetPreprocessedInputBatch()
        sess.run(tf.global_variables_initializer())
        batch_val = sess.run(batch)
        self.assertEqual(1, batch_val.src.source_features.shape[0])
        self.assertEqual(128, batch_val.src.source_features.shape[2])

  def testNlmsAecReducesEcho(self):
    rng = np.random.RandomState(0)
    num_samples = 4000
    reference = rng.normal(0.0, 0.2, size=num_samples).astype(np.float32)
    true_rir = np.array([0.8, -0.3, 0.15, -0.05], dtype=np.float32)
    echo = np.convolve(reference, true_rir, mode='full')[:num_samples]
    clean = 0.02 * np.sin(
        np.linspace(0.0, 40.0, num_samples, dtype=np.float32))
    mixed = clean + echo

    nlms = aec_model.NlmsAec(filter_length=16, step_size=0.5)
    error_sig, _, _ = nlms.process(mixed, reference)

    initial_mse = np.mean(np.square(mixed[2000:] - clean[2000:]))
    residual_mse = np.mean(np.square(error_sig[2000:] - clean[2000:]))
    self.assertLess(residual_mse, initial_mse * 0.1)


if __name__ == '__main__':
  tf.test.main()
