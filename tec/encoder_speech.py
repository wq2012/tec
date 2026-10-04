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
    inputs = batch.src_inputs
    if len(inputs.shape) == 3:
      inputs = tf.expand_dims(inputs, axis=-1)
    elif len(inputs.shape) == 4 and inputs.shape[-1] is None:
      dyn_shape = py_utils.GetShape(inputs)
      inputs = tf.reshape(
          inputs, [dyn_shape[0], dyn_shape[1], dyn_shape[2], 1])

    encoder_in = batch.DeepCopy()
    encoder_in.src_inputs = inputs
    return super().FProp(theta, encoder_in)
