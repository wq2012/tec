"""Text encoder for interfering TTS transcripts in Textual Echo Cancellation."""

from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import layers
from lingvo.core import py_utils
from lingvo.core import rnn_cell
from lingvo.core import rnn_layers


class TtsEncoderV2(base_layer.BaseLayer):
  """Text encoder for TTS character sequences (Table 1 of the paper).

  Architecture:
  - Character embedding lookup (`vocab_size=96`, `embedding_dim=512`).
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
            vocab_size=96,
            embedding_dim=512,
            scale_sqrt_depth=False,
            params_init=py_utils.WeightInit.Uniform(0.1)),
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
        zo_prob=p.zoneout_prob)
    bi_rnn_p = rnn_layers.BidirectionalFRNN.Params().Set(
        name='bi_lstm',
        fwd=cell_tpl.Copy().Set(name='fwd_lstm'),
        bak=cell_tpl.Copy().Set(name='bak_lstm'))
    self.CreateChild('bi_frnn', bi_rnn_p)

  def FProp(self, theta, input_batch):
    """Encodes token IDs `[batch, seq_len]` into `[seq_len, batch, 2 * cell]`.

    Args:
      theta: NestedMap of layer weights.
      input_batch: NestedMap with `ids` (`[batch, seq_len]`) and `paddings`
        (`[batch, seq_len]`).

    Returns:
      NestedMap with `encoded` (`[seq_len, batch, 2 * lstm_cell_size]`) and
      `padding` (`[seq_len, batch]`).
    """
    p = self.params
    with tf.name_scope(p.name):
      embedded = self.emb.EmbLookup(theta.emb, input_batch.ids)
      features_4d = tf.expand_dims(embedded, axis=2)
      paddings = input_batch.paddings

      for conv_layer, conv_theta in zip(self.conv, theta.conv):
        features_4d, paddings = conv_layer.FProp(
            conv_theta, features_4d, paddings)
        features_4d = self.dropout.FProp(theta.dropout, features_4d)

      batch_size, seq_len = py_utils.GetShape(features_4d, 2)
      channels = p.filter_shapes[-1][-1]
      features_3d = tf.reshape(features_4d, [batch_size, seq_len, channels])

      time_major = tf.transpose(features_3d, [1, 0, 2])
      pad_time_major = tf.transpose(paddings, [1, 0])
      encoded = self.bi_frnn.FProp(
          theta.bi_frnn, time_major, tf.expand_dims(pad_time_major, axis=-1))
      return py_utils.NestedMap(encoded=encoded, padding=pad_time_major)
