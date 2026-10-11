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
    p.Define(
        'input_residual_weight',
        1.0,
        'Residual weight for 4-frame grouped input spectrogram features.')
    p.Define(
        'active_energy_only',
        False,
        'If True, applies input residual only to log-Mel energy above silence.')
    return p

  def FProp(self, theta, batch):
    """Encodes a batch of log-Mel spectrograms into `[time, batch, 512]`."""
    p = self.params
    inputs = batch.src_inputs
    if len(inputs.shape) == 3:
      src_3d = inputs
      inputs = tf.expand_dims(inputs, axis=-1)
    elif len(inputs.shape) == 4 and inputs.shape[-1] is None:
      dyn_shape = py_utils.GetShape(inputs)
      inputs = tf.reshape(
          inputs, [dyn_shape[0], dyn_shape[1], dyn_shape[2], 1])
      src_3d = tf.squeeze(inputs, axis=-1)
    else:
      src_3d = tf.squeeze(inputs, axis=-1)

    encoder_in = batch.DeepCopy()
    encoder_in.src_inputs = inputs
    out = super().FProp(theta, encoder_in)

    out_steps = tf.shape(out.encoded)[0]
    batch_size = tf.shape(out.encoded)[1]
    out_dim = p.lstm_cell_size * 2
    feat_dim = p.input_shape[2] or 128
    stride_factor = 1
    for s in p.conv_filter_strides:
      stride_factor *= s[0]

    if p.input_residual_weight != 0.0 and out_dim <= feat_dim * stride_factor:
      residual_src = (
          tf.nn.relu(src_3d + 6.907755) if p.active_energy_only else src_3d)
      target_frames = out_steps * stride_factor
      in_frames = tf.shape(residual_src)[1]
      sliced = residual_src[:, :target_frames, :]
      rem = tf.maximum(0, target_frames - in_frames)
      padded = tf.pad(sliced, [[0, 0], [0, rem], [0, 0]])
      grouped = tf.reshape(
          padded, [batch_size, out_steps, feat_dim * stride_factor])
      grouped_tm = tf.transpose(grouped[:, :, :out_dim], [1, 0, 2])
      valid_tm = 1.0 - tf.expand_dims(out.padding, axis=-1)
      out.encoded = (
          out.encoded * 0.1 + p.input_residual_weight * grouped_tm) * valid_tm
    return out
