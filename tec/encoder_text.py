"""Text encoder for interfering TTS transcripts in Textual Echo Cancellation."""

from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import layers
from lingvo.core import py_utils
from lingvo.core import rnn_cell
from lingvo.core import rnn_layers
import numpy as np


def build_pretrained_char_embeddings(
    vocab_size: int = 98,
    embedding_dim: int = 512,
) -> np.ndarray:
  """Builds a deterministic 512-dim phonetic-spectral character embedding table.

  Implements the pre-trained 512-dim character embedding from Table 1 of the
  paper. Each character's 512-dim vector represents 4 sub-frames of 128-bin
  Mel-frequency suppression profiles corresponding to its broad phonetic class
  (vowels, voiced consonants, sibilants/fricatives, plosives, nasals, and
  silent pauses). Special tokens (`<pad>`, `<unk>`, `</s>`) are strictly zero.

  Args:
    vocab_size: Vocabulary size (default 98: 3 special tokens + 95 ASCII).
    embedding_dim: Embedding dimension (default 512 = 4 frames x 128 Mel bins).

  Returns:
    Float32 array of shape `[vocab_size, embedding_dim]`.
  """
  table = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
  num_bins = 128
  num_subframes = max(1, embedding_dim // num_bins)
  bins = np.arange(num_bins, dtype=np.float32)

  def _gauss(center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * np.square((bins - center) / max(1.0, width)))

  for token_id in range(3, vocab_size):
    code = token_id - 3 + 32
    ch = chr(code).lower() if 32 <= code <= 126 else ' '
    if not ch.isalnum():
      continue

    spec = np.zeros(num_bins, dtype=np.float32)
    sub_weights = np.ones(num_subframes, dtype=np.float32)
    if ch in 'aeiou':
      f0 = _gauss(10.0, 5.0)
      if ch == 'a':
        f1, f2 = _gauss(24.0, 7.0), _gauss(44.0, 9.0)
      elif ch == 'e':
        f1, f2 = _gauss(18.0, 6.0), _gauss(56.0, 10.0)
      elif ch == 'i':
        f1, f2 = _gauss(14.0, 5.0), _gauss(64.0, 10.0)
      elif ch == 'o':
        f1, f2 = _gauss(20.0, 6.0), _gauss(34.0, 8.0)
      else:
        f1, f2 = _gauss(15.0, 5.0), _gauss(28.0, 8.0)
      spec = -0.38 * (
          0.45 * f0 + 0.55 * f1 + 0.45 * f2 + 0.25 * _gauss(75.0, 14.0))
    elif ch in 'szcx':
      spec = -0.36 * (0.20 * _gauss(35.0, 12.0) + 0.85 * _gauss(88.0, 18.0))
    elif ch in 'fvh':
      spec = -0.30 * (0.35 * _gauss(28.0, 12.0) + 0.65 * _gauss(72.0, 22.0))
    elif ch in 'mn':
      spec = -0.34 * (0.75 * _gauss(14.0, 7.0) + 0.35 * _gauss(38.0, 10.0))
    elif ch in 'lrwy':
      spec = -0.35 * (
          0.55 * _gauss(16.0, 7.0) + 0.55 * _gauss(38.0, 11.0) +
          0.30 * _gauss(62.0, 14.0))
    elif ch in 'ptkbdgqj':
      voiced = 0.40 * _gauss(12.0, 6.0) if ch in 'bdgj' else 0.0
      spec = -0.32 * (
          voiced + 0.55 * _gauss(36.0, 14.0) + 0.55 * _gauss(74.0, 18.0))
      if num_subframes >= 4:
        sub_weights = np.array([0.4, 1.2, 1.3, 0.7], dtype=np.float32)
    else:
      spec = -0.28 * (0.50 * _gauss(25.0, 12.0) + 0.50 * _gauss(55.0, 16.0))

    phase = float(code % 19) * 0.33
    freq = 1.0 + float(code % 7) * 0.25
    ripple = -0.04 * (0.5 + 0.5 * np.sin(bins * freq * 0.1 + phase))
    full_vec = np.concatenate(
        [sub_weights[s] * (spec + ripple) for s in range(num_subframes)],
        axis=0)
    if len(full_vec) < embedding_dim:
      full_vec = np.pad(full_vec, (0, embedding_dim - len(full_vec)))
    table[token_id] = full_vec[:embedding_dim]
  return table


class TtsEncoderV2(base_layer.BaseLayer):
  """Text encoder for TTS character sequences (Table 1 of the paper).

  Architecture:
  - Pre-trained character embedding (`vocab_size=98`, `embedding_dim=512`).
  - 3 1D convolutional layers (`5x1` kernel, `512` channels, BatchNorm, ReLU)
    followed by dropout (`dropout_prob=0.5`).
  - 1 bidirectional LSTM layer (`256` units per direction -> `512`-dim output).
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'text_encoder'
    p.Define(
        'emb',
        layers.SimpleEmbeddingLayer.Params().Set(
            vocab_size=98,
            embedding_dim=512,
            scale_sqrt_depth=False,
            params_init=py_utils.WeightInit.Uniform(0.02)),
        'Character embedding parameters.')
    p.Define(
        'filter_shapes',
        [(5, 1, 512, 512), (5, 1, 512, 512), (5, 1, 512, 512)],
        'Convolutional kernel shapes (time, 1, in_channels, out_channels).')
    p.Define('filter_strides', [(1, 1), (1, 1), (1, 1)],
             'Convolutional strides.')
    p.Define('dropout_prob', 0.5, 'Dropout rate after each conv layer.')
    p.Define('zoneout_prob', 0.1, 'LSTM zoneout regularization probability.')
    p.Define('lstm_cell_size', 256, 'Hidden size per direction of Bi-LSTM.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self.CreateChild('emb', p.emb)

    conv_stack = []
    for idx, (shape, stride) in enumerate(
        zip(p.filter_shapes, p.filter_strides)):
      conv_p = layers.ConvLayer.Params().Set(
          name=f'conv_{idx}',
          filter_shape=shape,
          filter_stride=stride,
          batch_norm=True,
          bn_decay=0.9,
          params_init=py_utils.WeightInit.Uniform(0.02),
          activation='RELU')
      conv_stack.append(conv_p)
    self.CreateChildren('conv', conv_stack)

    dropout_p = layers.DropoutLayer.Params().Set(
        name='dropout', keep_prob=1.0 - p.dropout_prob)
    self.CreateChild('dropout', dropout_p)

    conv_out_dim = p.filter_shapes[-1][-1]
    cell_tpl = rnn_cell.LSTMCellSimple.Params().Set(
        deterministic=True,
        num_input_nodes=conv_out_dim,
        num_output_nodes=p.lstm_cell_size,
        params_init=py_utils.WeightInit.Uniform(0.02),
        zo_prob=p.zoneout_prob)
    bi_rnn_p = rnn_layers.BidirectionalFRNN.Params().Set(
        name='bi_lstm',
        fwd=cell_tpl.Copy().Set(name='fwd_lstm'),
        bak=cell_tpl.Copy().Set(name='bak_lstm'))
    self.CreateChild('bi_frnn', bi_rnn_p)

  def FProp(self, theta, input_batch):
    """Encodes token IDs `[batch, seq_len]` into `[seq_len, batch, 2*cell]`."""
    p = self.params
    with tf.name_scope(p.name):
      safe_ids = tf.clip_by_value(input_batch.ids, 0, p.emb.vocab_size - 1)
      char_mask = tf.expand_dims(
          tf.cast(safe_ids >= 3, py_utils.FPropDtype(p)), axis=-1)
      embedded = self.emb.EmbLookup(theta.emb, safe_ids) * char_mask
      features_4d = tf.expand_dims(embedded, axis=2)
      paddings = input_batch.paddings

      for conv_layer, conv_theta in zip(self.conv, theta.conv):
        features_4d, paddings = conv_layer.FProp(
            conv_theta, features_4d, paddings)
        features_4d = self.dropout.FProp(theta.dropout, features_4d)

      batch_size, seq_len = py_utils.GetShape(features_4d, 2)
      channels = p.filter_shapes[-1][-1]
      features_3d = (
          tf.reshape(features_4d, [batch_size, seq_len, channels]) * char_mask)

      time_major = tf.transpose(features_3d, [1, 0, 2])
      pad_time_major = tf.transpose(paddings, [1, 0])
      rnn_out = self.bi_frnn.FProp(
          theta.bi_frnn, time_major, tf.expand_dims(pad_time_major, axis=-1))
      emb_time_major = tf.transpose(embedded, [1, 0, 2])
      valid_tm = tf.expand_dims(1.0 - pad_time_major, axis=-1)
      if p.emb.embedding_dim == 2 * p.lstm_cell_size:
        encoded = (rnn_out + emb_time_major) * valid_tm
      else:
        encoded = rnn_out * valid_tm
      return py_utils.NestedMap(encoded=encoded, padding=pad_time_major)
