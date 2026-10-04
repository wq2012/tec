"""Convolutional building blocks for Textual Echo Cancellation."""

from lingvo.core import base_layer
from lingvo.core import layers as lingvo_layers


class PostEditConvNet(base_layer.BaseLayer):
  """Five-layer 1D convolutional residual Post-Net (Table 1 of the paper).

  Takes the initial Mel spectrogram predicted by the autoregressive decoder
  pre-net + LSTM stack and predicts a residual spectrogram refinement. Each
  layer uses a 5x1 kernel with batch normalization; the first N-1 layers use
  tanh activation and the final layer is linear.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'post_edit_conv_net'
    p.Define('feature_dims', 128, 'Number of Mel spectrogram bins.')
    p.Define('hidden_channels', 512, 'Number of channels in hidden convs.')
    p.Define('num_layers', 5, 'Number of 1D convolutional layers.')
    p.Define('kernel_size', 5, 'Temporal convolution kernel width.')
    p.Define('filter_shapes', None,
             'Optional explicit list of [kernel, 1, in_c, out_c] shapes.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if p.filter_shapes is not None:
      shapes = [list(s) for s in p.filter_shapes]
      if shapes[0][2] is None:
        shapes[0][2] = p.feature_dims
      if shapes[-1][3] is None:
        shapes[-1][3] = p.feature_dims
    else:
      shapes = []
      for idx in range(p.num_layers):
        in_c = p.feature_dims if idx == 0 else p.hidden_channels
        out_c = p.feature_dims if idx == p.num_layers - 1 else p.hidden_channels
        shapes.append([p.kernel_size, 1, in_c, out_c])

    conv_stack = []
    for idx, shape in enumerate(shapes):
      is_last = idx == len(shapes) - 1
      conv_p = lingvo_layers.ConvLayer.Params().Set(
          name=f'conv_{idx}',
          filter_shape=shape,
          filter_stride=(1, 1),
          batch_norm=True,
          activation='NONE' if is_last else 'TANH')
      conv_stack.append(conv_p)
    self.CreateChildren('conv', conv_stack)

  def FProp(self, theta, inputs, paddings):
    """Computes residual refinement of shape `[batch, time, 1, dim]`."""
    hidden = inputs
    for conv_layer, conv_theta in zip(self.conv, theta.conv):
      hidden, _ = conv_layer.FProp(conv_theta, hidden, paddings)
    return hidden
