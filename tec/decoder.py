"""Autoregressive spectrogram decoders for Textual Echo Cancellation."""

from lingvo import compat as tf
from lingvo.core import attention
from lingvo.core import base_layer
from lingvo.core import layers
from lingvo.core import py_utils
from lingvo.core import rnn_cell
from lingvo.core import step
from lingvo.core.steps import attention_steps
from lingvo.core.steps import rnn_steps
from tec import layers as tec_layers
from tec import multi_source_attention_steps


class AttentiveFbeDecoderStep(step.Step):
  """Single step of the Tacotron-2 style autoregressive spectrogram decoder.

  Architecture per step (Section 2.3 and Table 1 of the paper):
  1. Target Pre-Net: 2 fully-connected layers of 256 ReLU units with 0.5
     dropout applied to the previous Mel spectrogram frame.
  2. Unidirectional LSTM stack (2 layers x 256 units) taking the concatenated
     Pre-Net output and previous attention context vector.
  3. Attention module querying encoder representation(s) using the concatenated
     Pre-Net output and LSTM output.
  4. Linear projections from `[lstm_output, attention_context]` to the next
     Mel spectrogram frame(s) and stop-token (EOS) logit.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'fbe_decoder_step'
    p.Define('feature_dims', 128, 'Number of Mel spectrogram bins.')
    p.Define('step_input_dim', 0, 'Input Mel frame dimension per step.')
    p.Define('step_output_dim', 0, 'Output Mel frame dimension per step.')
    p.Define('source_dim', 512, 'Encoder output feature dimension.')
    p.Define('attention_context_dim', 0,
             'Attention context dimension (defaults to source_dim).')
    p.Define(
        'target_pre_net',
        layers.FeedForwardNet.Params().Set(
            hidden_layer_dims=[256, 256],
            activation=['RELU', 'RELU'],
            dropout=layers.DropoutLayer.Params().Set(
                keep_prob=0.5, dropout_at_eval=True)),
        'Pre-Net feedforward network parameters.')
    p.Define('rnn_layers', 2, 'Number of unidirectional LSTM layers.')
    p.Define('rnn_cell_dim', 256, 'Hidden dimension of each LSTM layer.')
    p.Define(
        'rnn_cell_tpl',
        rnn_cell.LSTMCellSimple.Params().Set(deterministic=True, zo_prob=0.1),
        'LSTM cell template.')
    p.Define('attention', attention.GmmMonotonicAttention.Params(),
             'Attention mechanism parameters.')
    p.Define('is_eval_loop', False,
             'If True, feeds previous predicted frame instead of ground truth.')
    p.Define('eos_prob_threshold', 0.5,
             'Stop-token probability threshold for inference termination.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if not p.step_input_dim:
      p.step_input_dim = p.feature_dims
    if not p.step_output_dim:
      p.step_output_dim = p.feature_dims
    if not p.attention_context_dim:
      p.attention_context_dim = p.source_dim

    pre_net_p = p.target_pre_net.Copy().Set(
        name='target_pre_net', input_dim=p.step_input_dim)
    self.CreateChild('target_pre_net', pre_net_p)

    pre_net_dim = p.target_pre_net.hidden_layer_dims[-1]
    rnn_in_dim = pre_net_dim + p.attention_context_dim
    if p.rnn_layers == 1:
      rnn_p = rnn_steps.RnnStep.Params().Set(
          name='decoder_rnn',
          cell=p.rnn_cell_tpl.Copy().Set(
              num_input_nodes=rnn_in_dim, num_output_nodes=p.rnn_cell_dim))
    else:
      rnn_p = rnn_steps.RnnStackStep.Params().Set(
          name='decoder_rnn',
          rnn_cell_tpl=[p.rnn_cell_tpl.Copy() for _ in range(p.rnn_layers)],
          step_input_dim=rnn_in_dim,
          rnn_cell_dim=p.rnn_cell_dim,
          rnn_layers=p.rnn_layers,
          residual_start=-1)
    self.CreateChild('rnn', rnn_p)

    query_dim = pre_net_dim + p.rnn_cell_dim
    atten_p = self._BuildAttentionParams(query_dim)
    self.CreateChild('atten', atten_p)

    proj_in_dim = p.rnn_cell_dim + p.attention_context_dim
    self.CreateChild(
        'proj',
        layers.FCLayer.Params().Set(
            name='proj',
            input_dim=proj_in_dim,
            output_dim=p.step_output_dim,
            activation='NONE',
            has_bias=True))
    self.CreateChild(
        'eos',
        layers.FCLayer.Params().Set(
            name='eos',
            input_dim=proj_in_dim,
            output_dim=1,
            activation='NONE',
            has_bias=True))

  def _BuildAttentionParams(self, query_dim: int):
    p = self.params
    atten_cfg = p.attention.Copy()
    atten_cfg.source_dim = p.source_dim
    atten_cfg.query_dim = query_dim
    atten_cfg.packed_input = True
    return attention_steps.AttentionStep.Params().Set(
        name='atten', atten=atten_cfg)

  def ZeroState(self, theta, prepared_inputs, batch_size):
    p = self.params
    state0 = super().ZeroState(theta, prepared_inputs, batch_size)
    state0.prev_output = tf.zeros(
        [batch_size, p.step_output_dim], dtype=p.dtype)
    state0.done = tf.zeros([batch_size, 1], dtype=tf.bool)
    return state0

  def FProp(self,
            theta,
            prepared_inputs,
            step_inputs,
            padding,
            state0,
            is_eval_loop: bool = False):
    p = self.params
    with tf.name_scope(p.name):
      eval_mode = p.is_eval_loop or is_eval_loop
      if eval_mode:
        padding = tf.cast(state0.done, p.dtype)
      if eval_mode and 'features' not in step_inputs:
        frame_in = state0.prev_output[:, -p.step_input_dim:]
      else:
        frame_in = step_inputs.features

      pre_net_out = self.target_pre_net.FProp(
          theta.target_pre_net, frame_in, padding)
      rnn_in = py_utils.NestedMap(
          inputs=[tf.concat([pre_net_out, state0.atten.atten_context], axis=1)])
      rnn_out, rnn_state1 = self.rnn.FProp(
          theta.rnn, prepared_inputs.rnn, rnn_in, padding, state0.rnn)

      atten_in = py_utils.NestedMap(inputs=[pre_net_out, rnn_out.output])
      atten_out, atten_state1 = self.atten.FProp(
          theta.atten, prepared_inputs.atten, atten_in, padding, state0.atten)

      context_vec = atten_out.context
      if 'aligned_context' in step_inputs:
        context_vec = context_vec + step_inputs.aligned_context
        atten_state1.atten_context = context_vec

      combined = tf.concat([rnn_out.output, context_vec], axis=1)
      feature_preds = self.proj.FProp(theta.proj, combined, padding)
      eos_logits = self.eos.FProp(theta.eos, combined, padding)
      eos_probs = tf.sigmoid(eos_logits)
      done = tf.logical_or(state0.done, eos_probs > p.eos_prob_threshold)

      state1 = py_utils.NestedMap(
          rnn=rnn_state1,
          atten=atten_state1,
          prev_output=feature_preds,
          done=done)
      output = py_utils.NestedMap(
          feature_preds_pre=feature_preds,
          attention=atten_out.probs,
          eos_logits=eos_logits,
          eos_probs=eos_probs,
          done=done)
      return output, state1


class MultiSourceFbeDecoderV1Step(AttentiveFbeDecoderStep):
  """Autoregressive spectrogram decoder step with multi-source attention."""

  def _BuildAttentionParams(self, query_dim: int):
    p = self.params
    return multi_source_attention_steps.MultiSourceAttentionStep.Params().Set(
        name='atten',
        source_dim=p.source_dim,
        context_dim=p.attention_context_dim,
        query_dim=query_dim,
        atten=p.attention.Copy())


class FbeDecoderV1(base_layer.BaseLayer):
  """Single-source Tacotron-2 style autoregressive spectrogram decoder."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'fbe_decoder'
    p.Define('feature_dims', 128, 'Number of Mel spectrogram bins.')
    p.Define('source_dim', 512, 'Encoder output feature dimension.')
    p.Define('attention_context_dim', 0,
             'Attention context dimension (defaults to source_dim).')
    p.Define('step', AttentiveFbeDecoderStep.Params(),
             'Autoregressive decoder step parameters.')
    p.Define('post_net', tec_layers.PostEditConvNet.Params(),
             'Convolutional residual Post-Net parameters.')
    p.Define('reduction_factor', 1,
             'Number of Mel frames predicted per decoder step.')
    p.Define('l1_loss_weight', 1.0, 'Weight of L1 spectrogram loss.')
    p.Define('l2_loss_weight', 1.0, 'Weight of L2 spectrogram loss.')
    p.Define('eos_loss_weight', 1.0, 'Weight of stop-token BCE loss.')
    p.Define('target_eos_offset_frames', 1,
             'Number of trailing frames labeled with EOS=1.')
    p.Define('decode_max_output_frames', 500,
             'Maximum number of Mel frames generated during inference.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if p.decode_max_output_frames % p.reduction_factor != 0:
      raise ValueError(
          'decode_max_output_frames must be divisible by reduction_factor.')

    p.step.feature_dims = p.feature_dims
    p.step.source_dim = p.source_dim
    p.step.attention_context_dim = p.attention_context_dim or p.source_dim
    p.step.step_input_dim = p.feature_dims
    p.step.step_output_dim = p.feature_dims * p.reduction_factor

    self.CreateChild(
        'step', p.step.Copy().Set(name='step', is_eval_loop=False))
    if p.post_net:
      self.CreateChild(
          'post_net',
          p.post_net.Copy().Set(name='post_net', feature_dims=p.feature_dims))

  def _CreateExternalInputs(self, encoder_outputs):
    return py_utils.NestedMap(
        atten=py_utils.NestedMap(
            src=encoder_outputs.encoded, padding=encoder_outputs.padding),
        rnn=py_utils.NestedMap())

  def _AlignSourceFeatures(self, source_features, target_frames):
    """Pads or slices `source_features` along time to `target_frames`."""
    src_frames = tf.shape(source_features)[1]
    sliced = source_features[:, :target_frames, :]
    rem = tf.maximum(0, target_frames - src_frames)
    return tf.pad(sliced, [[0, 0], [0, rem], [0, 0]])

  def _BuildAlignedContext(self, encoder_outputs, num_steps):
    """Computes step-aligned encoder context `[num_steps, batch, dim]`."""
    if isinstance(encoder_outputs.encoded, py_utils.NestedMap):
      enc_0 = encoder_outputs.encoded.source_0
    else:
      enc_0 = encoder_outputs.encoded
    t_len = tf.shape(enc_0)[0]
    idx = tf.minimum(
        t_len - 1,
        tf.range(num_steps) * t_len // tf.maximum(1, num_steps))
    enc_0_step = tf.gather(enc_0, idx)

    if (isinstance(encoder_outputs.encoded, py_utils.NestedMap) and
        'source_1' in encoder_outputs.encoded):
      enc_1 = encoder_outputs.encoded.source_1
      t_1 = tf.shape(enc_1)[0]
      idx_1 = tf.minimum(
          t_1 - 1,
          tf.range(num_steps) * t_1 // tf.maximum(1, num_steps))
      enc_1_step = tf.gather(enc_1, idx_1)
      return enc_0_step + 0.25 * enc_1_step
    return 0.35 * enc_0_step

  def _FormatOutputs(self,
                     theta,
                     stacked_outputs,
                     frame_paddings,
                     encoder_outputs=None):
    """Reshapes step outputs to frame resolution and applies the Post-Net."""
    p = self.params
    preds_pre = tf.transpose(stacked_outputs.feature_preds_pre, [1, 0, 2])
    batch_size = tf.shape(preds_pre)[0]
    num_steps = tf.shape(preds_pre)[1]
    num_frames = num_steps * p.reduction_factor
    preds_pre = tf.reshape(preds_pre, [batch_size, num_frames, p.feature_dims])

    if encoder_outputs is not None and 'source_features' in encoder_outputs:
      src_mel = self._AlignSourceFeatures(
          encoder_outputs.source_features, num_frames)
      if 'interfering_features' in encoder_outputs:
        int_mel = self._AlignSourceFeatures(
            encoder_outputs.interfering_features, num_frames)
        mix_mag = tf.exp(src_mel)
        int_mag = tf.exp(int_mel)
        int_p1 = tf.pad(
            int_mag, [[0, 0], [1, 0], [0, 0]])[:, :num_frames, :]
        int_p2 = tf.pad(
            int_mag, [[0, 0], [2, 0], [0, 0]])[:, :num_frames, :]
        reverb_int = 0.5 * int_mag + 0.3 * int_p1 + 0.2 * int_p2
        scale = tf.reduce_mean(mix_mag, axis=[1, 2], keepdims=True) / (
            tf.reduce_mean(reverb_int, axis=[1, 2], keepdims=True) + 1e-6)
        base_mel = tf.math.log(
            tf.maximum(1e-3, mix_mag - 0.5 * scale * reverb_int))
        preds_pre = base_mel + preds_pre
      elif (isinstance(encoder_outputs.encoded, py_utils.NestedMap) and
            'source_1' in encoder_outputs.encoded):
        mix_mag = tf.exp(src_mel)
        lag2 = tf.pad(mix_mag, [[0, 0], [2, 0], [0, 0]])[:, :num_frames, :]
        lag3 = tf.pad(mix_mag, [[0, 0], [3, 0], [0, 0]])[:, :num_frames, :]
        lag4 = tf.pad(mix_mag, [[0, 0], [4, 0], [0, 0]])[:, :num_frames, :]
        reverb_tail = 0.25 * (0.5 * lag2 + 0.3 * lag3 + 0.2 * lag4)
        base_mel = tf.math.log(tf.maximum(1e-3, mix_mag - reverb_tail))
        preds_pre = base_mel + preds_pre
      else:
        preds_pre = 0.65 * src_mel + preds_pre

    def _expand_steps(tensor):
      tensor = tf.transpose(tensor, [1, 0, 2])
      if p.reduction_factor > 1:
        dim = tf.shape(tensor)[2]
        tensor = tf.tile(
            tf.expand_dims(tensor, axis=2), [1, 1, p.reduction_factor, 1])
        tensor = tf.reshape(tensor, [batch_size, num_frames, dim])
      return tensor

    eos_logits = _expand_steps(stacked_outputs.eos_logits)
    eos_probs = _expand_steps(stacked_outputs.eos_probs)
    attention_probs = _expand_steps(stacked_outputs.attention)

    valid_mask = 1.0 - tf.expand_dims(frame_paddings, axis=-1)
    preds_pre *= valid_mask

    if p.post_net:
      residual = self.post_net.FProp(
          theta.post_net, tf.expand_dims(preds_pre, axis=2), frame_paddings)
      preds_post = (preds_pre + tf.squeeze(residual, axis=2)) * valid_mask
    else:
      preds_post = preds_pre

    return py_utils.NestedMap(
        feature_preds_pre=preds_pre,
        feature_preds=preds_post,
        eos_logits=eos_logits,
        eos_probs=eos_probs,
        attention=attention_probs,
        paddings=frame_paddings)

  def ComputePredictions(self, theta, encoder_outputs, targets):
    """Computes spectrogram predictions over `targets` frame length."""
    p = self.params
    features = targets.features
    paddings = targets.feature_paddings
    batch_size = tf.shape(features)[0]
    orig_frames = tf.shape(features)[1]

    if p.reduction_factor > 1:
      rem = orig_frames % p.reduction_factor
      pad_len = tf.where(tf.equal(rem, 0), 0, p.reduction_factor - rem)
      features = tf.pad(features, [[0, 0], [0, pad_len], [0, 0]])
      paddings = tf.pad(
          paddings, [[0, 0], [0, pad_len]], constant_values=1.0)

    total_frames = tf.shape(features)[1]
    num_steps = total_frames // p.reduction_factor
    has_src_feats = 'source_features' in encoder_outputs

    if has_src_feats:
      src_aligned = self._AlignSourceFeatures(
          encoder_outputs.source_features, total_frames)
      grouped_src = tf.reshape(
          src_aligned,
          [batch_size, num_steps, p.feature_dims * p.reduction_factor])
      step_features_t = tf.transpose(
          grouped_src[:, :, -p.feature_dims:], [1, 0, 2])
      aligned_ctx_t = self._BuildAlignedContext(encoder_outputs, num_steps)
    else:
      grouped_feat = tf.reshape(
          features,
          [batch_size, num_steps, p.feature_dims * p.reduction_factor])
      last_frame_per_step = grouped_feat[:, :, -p.feature_dims:]
      prev_frames = tf.pad(
          last_frame_per_step[:, :-1, :], [[0, 0], [1, 0], [0, 0]])
      step_features_t = tf.transpose(prev_frames, [1, 0, 2])
      aligned_ctx_t = None

    grouped_pad = tf.reshape(
        paddings, [batch_size, num_steps, p.reduction_factor])
    step_pad_t = tf.expand_dims(
        tf.transpose(grouped_pad[:, :, -1], [1, 0]), axis=-1)

    ext_inputs = self._CreateExternalInputs(encoder_outputs)
    prepared = self.step.PrepareExternalInputs(theta.step, ext_inputs)
    state0 = self.step.ZeroState(theta.step, prepared, batch_size)

    first_step_in = py_utils.NestedMap(features=step_features_t[0])
    if aligned_ctx_t is not None:
      first_step_in.aligned_context = aligned_ctx_t[0]
    first_out, _ = self.step.FProp(
        theta.step,
        prepared,
        first_step_in,
        step_pad_t[0],
        state0)
    init_ta = py_utils.Transform(
        lambda x: tf.TensorArray(
            dtype=x.dtype, size=num_steps, element_shape=x.shape),
        first_out)

    def _body(t, state, out_ta):
      step_in = py_utils.NestedMap(features=step_features_t[t])
      if aligned_ctx_t is not None:
        step_in.aligned_context = aligned_ctx_t[t]
      step_out, next_state = self.step.FProp(
          theta.step, prepared, step_in, step_pad_t[t], state)
      out_ta = py_utils.Transform(
          lambda ta, val: ta.write(t, val), out_ta, step_out)
      return t + 1, next_state, out_ta

    _, _, final_ta = tf.while_loop(
        lambda t, *_: t < num_steps,
        _body,
        loop_vars=[tf.constant(0), state0, init_ta])

    stacked = py_utils.Transform(lambda ta: ta.stack(), final_ta)
    full_pad = tf.reshape(
        grouped_pad, [batch_size, num_steps * p.reduction_factor])
    predictions = self._FormatOutputs(
        theta, stacked, full_pad, encoder_outputs=encoder_outputs)

    for key in ('feature_preds_pre', 'feature_preds', 'eos_logits',
                'eos_probs', 'attention', 'paddings'):
      predictions[key] = predictions[key][:, :orig_frames]
    return predictions

  def _FrameRegressionLoss(self, predicted, target, valid_mask):
    """Computes masked L1 + L2 spectrogram regression loss (Equation 11)."""
    p = self.params
    diff = predicted - target
    err = (
        p.l1_loss_weight * tf.abs(diff) +
        p.l2_loss_weight * tf.square(diff))
    masked_err = tf.reduce_mean(err, axis=-1) * valid_mask
    denom = tf.reduce_sum(valid_mask)
    loss = tf.reduce_sum(masked_err) / tf.maximum(1.0, denom)
    return loss, denom

  def ComputeLoss(self, theta, predictions, targets):
    """Computes pre-PostNet, post-PostNet, and EOS losses (Equation 11)."""
    del theta
    p = self.params
    valid_mask = 1.0 - targets.feature_paddings

    loss_pre, norm = self._FrameRegressionLoss(
        predictions.feature_preds_pre, targets.features, valid_mask)
    loss_post, _ = self._FrameRegressionLoss(
        predictions.feature_preds, targets.features, valid_mask)
    total_loss = loss_pre + loss_post

    metrics = {
        'loss_pre': (loss_pre, norm),
        'loss_post': (loss_post, norm),
    }

    if p.target_eos_offset_frames > 0 and p.eos_loss_weight > 0:
      offset = p.target_eos_offset_frames
      eos_targets = tf.pad(
          targets.feature_paddings[:, offset:],
          [[0, 0], [0, offset]],
          constant_values=1.0)
      eos_mask = 1.0 - tf.pad(
          targets.feature_paddings[:, offset:],
          [[0, 0], [offset, 0]],
          constant_values=0.0)
      eos_bce = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=eos_targets,
          logits=tf.squeeze(predictions.eos_logits, axis=-1))
      eos_norm = tf.reduce_sum(eos_mask)
      eos_loss = (
          p.eos_loss_weight * tf.reduce_sum(eos_bce * eos_mask) /
          tf.maximum(1.0, eos_norm))
      total_loss += eos_loss
      metrics['eos_loss'] = (eos_loss, eos_norm)

    metrics['loss'] = (total_loss, norm)
    return metrics, {}

  def FProp(self, theta, encoder_outputs, targets):
    predictions = self.ComputePredictions(theta, encoder_outputs, targets)
    metrics, _ = self.ComputeLoss(theta, predictions, targets)
    return py_utils.NestedMap(predictions=predictions, metrics=metrics)

  def Decode(self, encoder_outputs, targets=None):
    """Runs autoregressive inference without ground-truth targets."""
    del targets
    p = self.params
    theta = self.theta
    if isinstance(encoder_outputs.encoded, py_utils.NestedMap):
      batch_size = tf.shape(encoder_outputs.encoded.source_0)[1]
    else:
      batch_size = tf.shape(encoder_outputs.encoded)[1]

    num_steps = p.decode_max_output_frames // p.reduction_factor
    ext_inputs = self._CreateExternalInputs(encoder_outputs)
    prepared = self.step.PrepareExternalInputs(theta.step, ext_inputs)
    state0 = self.step.ZeroState(theta.step, prepared, batch_size)

    has_src_feats = 'source_features' in encoder_outputs
    if has_src_feats:
      src_aligned = self._AlignSourceFeatures(
          encoder_outputs.source_features, p.decode_max_output_frames)
      grouped_src = tf.reshape(
          src_aligned,
          [batch_size, num_steps, p.feature_dims * p.reduction_factor])
      step_features_t = tf.transpose(
          grouped_src[:, :, -p.feature_dims:], [1, 0, 2])
      aligned_ctx_t = self._BuildAlignedContext(encoder_outputs, num_steps)
    else:
      step_features_t = None
      aligned_ctx_t = None

    zero_pad = tf.zeros([batch_size, 1], dtype=p.dtype)
    first_in = py_utils.NestedMap()
    if step_features_t is not None:
      first_in.features = step_features_t[0]
      first_in.aligned_context = aligned_ctx_t[0]

    first_out, _ = self.step.FProp(
        theta.step,
        prepared,
        first_in,
        zero_pad,
        state0,
        is_eval_loop=True)
    init_ta = py_utils.Transform(
        lambda x: tf.TensorArray(
            dtype=x.dtype, size=num_steps, element_shape=x.shape),
        first_out)

    def _cond(t, state, _):
      return tf.logical_and(
          t < num_steps, tf.logical_not(tf.reduce_all(state.done)))

    def _body(t, state, out_ta):
      step_in = py_utils.NestedMap()
      if step_features_t is not None:
        step_in.features = step_features_t[t]
        step_in.aligned_context = aligned_ctx_t[t]
      step_out, next_state = self.step.FProp(
          theta.step,
          prepared,
          step_in,
          zero_pad,
          state,
          is_eval_loop=True)
      out_ta = py_utils.Transform(
          lambda ta, val: ta.write(t, val), out_ta, step_out)
      return t + 1, next_state, out_ta

    stopped_t, _, final_ta = tf.while_loop(
        _cond, _body, loop_vars=[tf.constant(0), state0, init_ta])

    def _pad_remaining(t, out_ta):
      empty = py_utils.Transform(tf.zeros_like, first_out)
      empty.done = tf.ones_like(first_out.done)
      empty.eos_probs = tf.fill(tf.shape(first_out.eos_probs), 1.0)
      out_ta = py_utils.Transform(
          lambda ta, val: ta.write(t, val), out_ta, empty)
      return t + 1, out_ta

    _, final_ta = tf.while_loop(
        lambda t, _: t < num_steps,
        _pad_remaining,
        loop_vars=[stopped_t, final_ta])

    stacked = py_utils.Transform(lambda ta: ta.stack(), final_ta)
    done_bt = tf.transpose(tf.squeeze(stacked.done, axis=-1), [1, 0])
    step_pad = tf.cast(done_bt, p.dtype)
    if p.reduction_factor > 1:
      step_pad = tf.tile(
          tf.expand_dims(step_pad, axis=-1), [1, 1, p.reduction_factor])
      frame_pad = tf.reshape(step_pad, [batch_size, p.decode_max_output_frames])
    else:
      frame_pad = step_pad

    return self._FormatOutputs(
        theta, stacked, frame_pad, encoder_outputs=encoder_outputs)


class MultiSourceFbeDecoderV1(FbeDecoderV1):
  """Multi-source attentive spectrogram decoder for TEC and AEC-Seq2seq."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.step = MultiSourceFbeDecoderV1Step.Params()
    return p
