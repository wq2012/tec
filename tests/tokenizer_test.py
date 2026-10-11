"""Tests for tec.tokenizer."""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tec import tokenizer  # noqa: E402


class TokenizerTest(unittest.TestCase):

  def testTextToIdsRoundTrip(self):
    tok = tokenizer.CharTokenizer()
    self.assertEqual(98, tok.vocab_size)
    text = 'Currently in Mountain View, it is 72 degrees. {|}~<>\\`'
    ids = tok.text_to_ids(text)
    self.assertEqual(tok.eos_id, ids[-1])
    self.assertTrue(all(0 <= i < 98 for i in ids))
    self.assertNotIn(tok.unk_id, ids)
    decoded = tok.ids_to_text(ids)
    self.assertEqual(text, decoded)

  def testAllPrintableAsciiRoundTrip(self):
    tok = tokenizer.CharTokenizer()
    all_ascii = ''.join(chr(c) for c in range(32, 127))
    ids = tok.text_to_ids(all_ascii, append_eos=False)
    self.assertEqual(95, len(ids))
    self.assertEqual(95, len(set(ids)))
    self.assertNotIn(tok.unk_id, ids)
    self.assertEqual(all_ascii, tok.ids_to_text(ids))

  def testBatchEncodePadding(self):
    tok = tokenizer.CharTokenizer()
    ids, paddings = tok.batch_encode(['hi', 'hello world'])
    self.assertEqual((2, 12), ids.shape)
    self.assertEqual((2, 12), paddings.shape)
    np.testing.assert_allclose([0.0] * 3 + [1.0] * 9, paddings[0])
    np.testing.assert_allclose([0.0] * 12, paddings[1])


if __name__ == '__main__':
  unittest.main()
