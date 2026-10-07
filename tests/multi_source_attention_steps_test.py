"""Tests for tec.multi_source_attention_steps."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tec import multi_source_attention_steps  # noqa: E402
from lingvo import compat as tf  # noqa: E402
from lingvo.core import attention  # noqa: E402
from lingvo.core import py_utils  # noqa: E402
from lingvo.core import test_utils  # noqa: E402


class MultiSourceAttentionStepsTest(test_utils.TestCase):

  def _run_attention_step_check(self, atten_tpl):
    with self.session(graph=tf.Graph(), use_gpu=False) as sess:
      batch_size = 3
      seq_len_0 = 6
      seq_len_1 = 4
      source_dim = 8
      query_dim = 10

      src_0 = tf.random.normal([seq_len_0, batch_size, source_dim], seed=1)
      src_1 = tf.random.normal([seq_len_1, batch_size, source_dim], seed=2)
      pad_0 = tf.zeros([seq_len_0, batch_size], dtype=tf.float32)
      pad_1 = tf.zeros([seq_len_1, batch_size], dtype=tf.float32)
      query = tf.random.normal([batch_size, query_dim], seed=3)

      p = multi_source_attention_steps.MultiSourceAttentionStep.Params().Set(
          source_dim=source_dim,
          query_dim=query_dim,
          atten=atten_tpl)
      step = p.Instantiate()

      ext = py_utils.NestedMap(
          src=py_utils.NestedMap(source_0=src_0, source_1=src_1),
          padding=py_utils.NestedMap(source_0=pad_0, source_1=pad_1))
      packed = step.PrepareExternalInputs(step.theta, ext)
      state0 = step.ZeroState(step.theta, packed, batch_size)

      step_in = py_utils.NestedMap(inputs=[query])
      step_pad = tf.zeros([batch_size, 1], dtype=tf.float32)
      out, state1 = step.FProp(step.theta, packed, step_in, step_pad, state0)

      sess.run(tf.global_variables_initializer())
      out_val, state1_val = sess.run([out, state1])

      self.assertEqual((batch_size, source_dim), out_val.context.shape)
      self.assertEqual((batch_size, seq_len_0), out_val.probs.shape)
      self.assertTrue(np.all(out_val.probs >= 0.0))
      np.testing.assert_allclose(out_val.context, state1_val.atten_context)

  def testAdditiveMultiSourceAttention(self):
    atten_tpl = attention.AdditiveAttention.Params().Set(hidden_dim=8)
    self._run_attention_step_check(atten_tpl)

  def testGmmMonotonicMultiSourceAttention(self):
    atten_tpl = attention.GmmMonotonicAttention.Params().Set(
        hidden_dim=8, num_mixtures=3)
    self._run_attention_step_check(atten_tpl)


if __name__ == '__main__':
  tf.test.main()
