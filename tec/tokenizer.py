"""Character tokenizer for TTS text side input in Textual Echo Cancellation."""

from typing import List, Optional, Sequence, Tuple
import numpy as np

_SPECIAL_TOKENS = ('<pad>', '</s>', '<unk>')
_PRINTABLE_ASCII = (
    ' !"\'(),-.:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    '0123456789@#$%&*+/=_~^[]{}|<>\\`'
)


class CharTokenizer:
  """Maps TTS text strings to integer token IDs and padding masks."""

  def __init__(self, vocab_size: int = 96):
    self.vocab_size = vocab_size
    tokens = list(_SPECIAL_TOKENS) + list(_PRINTABLE_ASCII)
    while len(tokens) < vocab_size:
      tokens.append(f'<extra_{len(tokens)}>')
    self._id_to_token = tokens[:vocab_size]
    self._token_to_id = {tok: idx for idx, tok in enumerate(self._id_to_token)}
    self.pad_id = 0
    self.eos_id = 1
    self.unk_id = 2

  def text_to_ids(self, text: str, append_eos: bool = True) -> List[int]:
    """Encodes a string into a list of character token IDs."""
    if isinstance(text, bytes):
      text = text.decode('utf-8', errors='replace')
    ids = [self._token_to_id.get(ch, self.unk_id) for ch in text]
    if append_eos:
      ids.append(self.eos_id)
    return ids

  def ids_to_text(self, ids: Sequence[int]) -> str:
    """Decodes a sequence of character token IDs back into a string."""
    output = []
    for raw_id in ids:
      idx = int(raw_id)
      if idx in (self.pad_id, self.eos_id):
        continue
      if 0 <= idx < len(self._id_to_token):
        token = self._id_to_token[idx]
        if not token.startswith('<'):
          output.append(token)
    return ''.join(output)

  def batch_encode(
      self,
      texts: Sequence[str],
      max_length: Optional[int] = None,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Encodes a batch of strings into `(ids, paddings)` numpy arrays."""
    sequences = [self.text_to_ids(t) for t in texts]
    if max_length is None:
      max_length = max((len(seq) for seq in sequences), default=1)
    max_length = max(1, max_length)

    batch_size = len(texts)
    ids = np.full((batch_size, max_length), self.pad_id, dtype=np.int32)
    paddings = np.ones((batch_size, max_length), dtype=np.float32)

    for row, seq in enumerate(sequences):
      valid = min(len(seq), max_length)
      ids[row, :valid] = seq[:valid]
      paddings[row, :valid] = 0.0
    return ids, paddings
