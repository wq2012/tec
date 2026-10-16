"""Audio encoder for Textual Echo Cancellation."""

from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.tasks.asr import encoder as asr_encoder


class SpeechEncoderV1(asr_encoder.AsrEncoder):
  """Audio encoder for mixed and reference speech spectrograms (Table 1).

  Architecture:
  - 2 strided 2D convolutional layers with 3x3 kernels, 2x2 strides, and 32
    channels (4x time-frequency reduction).
  - 1 bidirectional convolutional LSTM (Bi-CLSTM) layer with 1x3 kernel.
  - 3 bidirectional LSTM layers with 256 hidden units per direction (512-dim
    bidirectional output).
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'speech_encoder'
    p.use_specaugment = False
    p.input_shape = [None, None, 128, 1]
    p.conv_filter_shapes = [(3, 3, 1, 32), (3, 3, 32, 32)]
    p.conv_filter_strides = [(2, 2), (2, 2)]
    p.num_conv_lstm_layers = 1
    p.num_lstm_layers = 3
    p.lstm_cell_size = 256
    p.project_lstm_output = False
    p.pad_steps = 0
    return p

  def FProp(self, theta, batch):
    """Encodes a batch of log-Mel spectrograms into `[time, batch, 512]`."""
    p = self.params
    inputs = batch.src_inputs
    paddings = batch.paddings
    if len(inputs.shape) == 3:
      inputs = tf.expand_dims(inputs, axis=-1)
    elif len(inputs.shape) == 4 and inputs.shape[-1] is None:
      dyn_shape = py_utils.GetShape(inputs)
      inputs = tf.reshape(
          inputs, [dyn_shape[0], dyn_shape[1], dyn_shape[2], 1])

    with tf.name_scope(p.name):
      conv_out = inputs
      out_padding = paddings
      for i, conv_layer in enumerate(self.conv):
        conv_out, out_padding = conv_layer.FProp(
            theta.conv[i], conv_out, out_padding)

      def _transpose_first_two_dims(t):
        first_dim = tf.shape(t)[0]
        second_dim = tf.shape(t)[1]
        t_new = tf.transpose(
            tf.reshape(t, [first_dim, second_dim, -1]), [1, 0, 2])
        t_shape_new = tf.concat([[second_dim], [first_dim], tf.shape(t)[2:]], 0)
        return tf.reshape(t_new, t_shape_new)

      conv_lstm_out = conv_out
      conv_lstm_out_padding = out_padding
      for i, (rnn, cnn) in enumerate(
          zip(self.conv_lstm_rnn, self.conv_lstm_cnn)):
        conv_lstm_in = _transpose_first_two_dims(conv_lstm_out)
        conv_lstm_in = tf.expand_dims(conv_lstm_in, 2)
        conv_lstm_in_padding = tf.expand_dims(
            tf.transpose(conv_lstm_out_padding), 2)
        lstm_out = rnn.FProp(
            theta.conv_lstm_rnn[i], conv_lstm_in, conv_lstm_in_padding)
        cnn_in = tf.squeeze(_transpose_first_two_dims(lstm_out), 2)
        cnn_out, cnn_out_padding = cnn.FProp(
            theta.conv_lstm_cnn[i], cnn_in, conv_lstm_out_padding)
        conv_lstm_out = conv_lstm_out + 0.02 * cnn_out
        conv_lstm_out_padding = cnn_out_padding

      conv_lstm_out_shape = tf.shape(conv_lstm_out)
      new_shape = tf.concat([conv_lstm_out_shape[:2], [-1]], 0)
      conv_lstm_out = tf.reshape(conv_lstm_out, new_shape)
      if self._first_lstm_input_dim_pad:
        conv_lstm_out = tf.pad(
            conv_lstm_out,
            [[0, 0], [0, 0], [0, self._first_lstm_input_dim_pad]])
      conv_lstm_out = py_utils.HasShape(
          conv_lstm_out, [-1, -1, self._first_lstm_input_dim])

      rnn_in = tf.transpose(conv_lstm_out, [1, 0, 2])
      rnn_padding = tf.expand_dims(tf.transpose(conv_lstm_out_padding), 2)
      for i in range(p.num_lstm_layers):
        rnn_out = self.rnn[i].FProp(theta.rnn[i], rnn_in, rnn_padding)
        if i == p.num_lstm_layers - 1:
          rnn_out *= (1.0 - rnn_padding)
        rnn_in = rnn_out

      return py_utils.NestedMap(
          encoded=rnn_in,
          padding=tf.squeeze(rnn_padding, [2]),
          state=py_utils.NestedMap())
