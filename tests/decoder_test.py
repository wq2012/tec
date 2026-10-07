"""Tests for tec.decoder."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tec import decoder as tec_decoder  # noqa: E402
from lingvo import compat as tf  # noqa: E402
from lingvo.core import attention  # noqa: E402
from lingvo.core import py_utils  # noqa: E402
from lingvo.core import test_utils  # noqa: E402


def _make_dummy_inputs(
    source_dim: int = 8,
    feature_dims: int = 4,
    batch_size: int = 2,
    src_len: int = 6,
    tgt_len: int = 5,
    single_source: bool = False,
):
  enc_0 = tf.random.normal([src_len, batch_size, source_dim], seed=1)
  pad_0 = tf.zeros([src_len, batch_size], dtype=tf.float32)
  if single_source:
    encoder_outputs = py_utils.NestedMap(encoded=enc_0, padding=pad_0)
  else:
    enc_1 = tf.random.normal([src_len, batch_size, source_dim], seed=2)
    pad_1 = tf.zeros([src_len, batch_size], dtype=tf.float32)
    encoder_outputs = py_utils.NestedMap(
        encoded=py_utils.NestedMap(source_0=enc_0, source_1=enc_1),
        padding=py_utils.NestedMap(source_0=pad_0, source_1=pad_1))

  targets = py_utils.NestedMap(
      features=tf.random.normal([batch_size, tgt_len, feature_dims], seed=3),
      feature_paddings=tf.zeros([batch_size, tgt_len], dtype=tf.float32))
  return encoder_outputs, targets


class DecoderTest(test_utils.TestCase):

  def _build_decoder_params(self, single_source: bool = False):
    cls = (
        tec_decoder.FbeDecoderV1
        if single_source else tec_decoder.MultiSourceFbeDecoderV1)
    p = cls.Params()
    p.source_dim = 8
    p.feature_dims = 4
    p.decode_max_output_frames = 6
    p.step.rnn_cell_dim = 8
    p.step.rnn_layers = 2
    p.step.target_pre_net.hidden_layer_dims = [8, 8]
    p.step.attention = attention.GmmMonotonicAttention.Params().Set(
        hidden_dim=8, num_mixtures=2)
    p.post_net.hidden_channels = 8
    p.post_net.num_layers = 2
    return p

  def testMultiSourceFPropAndDecode(self):
    p = self._build_decoder_params(single_source=False)
    with self.session(graph=tf.Graph(), use_gpu=False) as sess:
      encoder_outputs, targets = _make_dummy_inputs(single_source=False)
      dec = p.Instantiate()
      fprop_out = dec.FPropDefaultTheta(encoder_outputs, targets)
      dec_out = dec.Decode(encoder_outputs)
      sess.run(tf.global_variables_initializer())
      metrics, preds, decoded = sess.run(
          [fprop_out.metrics, fprop_out.predictions, dec_out])
      self.assertIn('loss_pre', metrics)
      self.assertIn('loss_post', metrics)
      self.assertIn('eos_loss', metrics)
      self.assertIn('loss', metrics)
      self.assertEqual((2, 5, 4), preds.feature_preds.shape)
      self.assertEqual((2, 6, 4), decoded.feature_preds.shape)

  def testSingleSourceFPropAndDecode(self):
    p = self._build_decoder_params(single_source=True)
    with self.session(graph=tf.Graph(), use_gpu=False) as sess:
      encoder_outputs, targets = _make_dummy_inputs(single_source=True)
      dec = p.Instantiate()
      fprop_out = dec.FPropDefaultTheta(encoder_outputs, targets)
      dec_out = dec.Decode(encoder_outputs)
      sess.run(tf.global_variables_initializer())
      preds, decoded = sess.run([fprop_out.predictions, dec_out])
      self.assertEqual((2, 5, 4), preds.feature_preds.shape)
      self.assertEqual((2, 6, 4), decoded.feature_preds.shape)

  def testReductionFactorAndEarlyStopping(self):
    p = self._build_decoder_params(single_source=False)
    p.decode_max_output_frames = 4
    p.reduction_factor = 2
    p.target_eos_offset_frames = 2
    p.step.eos_prob_threshold = 0.0
    with self.session(graph=tf.Graph(), use_gpu=False) as sess:
      encoder_outputs, targets = _make_dummy_inputs(
          tgt_len=4, single_source=False)
      dec = p.Instantiate()
      fprop_out = dec.FPropDefaultTheta(encoder_outputs, targets)
      dec_out = dec.Decode(encoder_outputs)
      sess.run(tf.global_variables_initializer())
      preds, decoded = sess.run([fprop_out.predictions, dec_out])
      self.assertEqual((2, 4, 4), preds.feature_preds.shape)
      self.assertEqual((2, 4, 4), decoded.feature_preds.shape)


if __name__ == '__main__':
  tf.test.main()
