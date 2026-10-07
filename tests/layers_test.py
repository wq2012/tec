"""Tests for tec.layers, tec.encoder_speech, and tec.encoder_text."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tec import encoder_speech  # noqa: E402
from tec import encoder_text  # noqa: E402
from tec import layers as tec_layers  # noqa: E402
from lingvo import compat as tf  # noqa: E402
from lingvo.core import py_utils  # noqa: E402
from lingvo.core import test_utils  # noqa: E402


class LayersAndEncodersTest(test_utils.TestCase):

  def testPostEditConvNetDefaultAndCustomShapes(self):
    with self.session(use_gpu=False) as sess:
      p = tec_layers.PostEditConvNet.Params().Set(
          name='post_net',
          feature_dims=16,
          hidden_channels=32,
          num_layers=3)
      layer = p.Instantiate()
      inputs = tf.random.normal([2, 6, 1, 16], seed=10)
      paddings = tf.zeros([2, 6], dtype=tf.float32)
      out = layer.FPropDefaultTheta(inputs, paddings)
      sess.run(tf.global_variables_initializer())
      out_val = sess.run(out)
      self.assertEqual((2, 6, 1, 16), out_val.shape)

  def testSpeechEncoderV1(self):
    with self.session(use_gpu=False) as sess:
      p = encoder_speech.SpeechEncoderV1.Params().Set(
          name='speech_enc',
          input_shape=[None, None, 16, 1],
          conv_filter_shapes=[(3, 3, 1, 8), (3, 3, 8, 8)],
          conv_filter_strides=[(2, 2), (2, 2)],
          num_conv_lstm_layers=1,
          num_lstm_layers=2,
          lstm_cell_size=16)
      enc = p.Instantiate()
      batch = py_utils.NestedMap(
          src_inputs=tf.random.normal([2, 12, 16], seed=20),
          paddings=tf.zeros([2, 12], dtype=tf.float32))
      out = enc.FPropDefaultTheta(batch)
      sess.run(tf.global_variables_initializer())
      enc_val, pad_val = sess.run([out.encoded, out.padding])
      self.assertEqual((3, 2, 32), enc_val.shape)
      self.assertEqual((3, 2), pad_val.shape)

  def testTtsEncoderV2(self):
    with self.session(use_gpu=False) as sess:
      p = encoder_text.TtsEncoderV2.Params().Set(
          name='text_enc',
          filter_shapes=[(3, 1, 32, 32), (3, 1, 32, 32)],
          filter_strides=[(1, 1), (1, 1)],
          lstm_cell_size=16)
      p.emb.vocab_size = 32
      p.emb.embedding_dim = 32
      enc = p.Instantiate()
      batch = py_utils.NestedMap(
          ids=tf.ones([2, 7], dtype=tf.int32),
          paddings=tf.zeros([2, 7], dtype=tf.float32))
      out = enc.FPropDefaultTheta(batch)
      sess.run(tf.global_variables_initializer())
      enc_val, pad_val = sess.run([out.encoded, out.padding])
      self.assertEqual((7, 2, 32), enc_val.shape)
      self.assertEqual((7, 2), pad_val.shape)


if __name__ == '__main__':
  tf.test.main()
