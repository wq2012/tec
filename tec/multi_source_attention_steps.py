"""Multi-source attention step for Textual Echo Cancellation."""

from lingvo import compat as tf
from lingvo.core import attention
from lingvo.core import py_utils
from lingvo.core import step


class MultiSourceAttentionStep(step.Step):
  """Attends jointly to the speech encoder and side-input encoder streams.

  Implements the multi-source attention mechanism described in Section 2.2
  (Equations 4-9) of the Textual Echo Cancellation paper
  (https://arxiv.org/pdf/2008.06006), combining the context vector from the
  noisy speech encoder (`source_0`) with the context vector from the TTS text
  or reference audio encoder (`source_1`).
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'multi_source_attention_step'
    p.Define('num_source', 2, 'Number of encoder sources to attend to.')
    p.Define('primary_source_index', 0,
             'Index of primary encoder stream for attention alignment return.')
    p.Define('source_dim', 512, 'Feature dimension of each encoder stream.')
    p.Define('context_dim', 0,
             'Output context vector dimension (defaults to source_dim).')
    p.Define('query_dim', 512, 'Dimension of the decoder query vector.')
    p.Define('atten', attention.GmmMonotonicAttention.Params(),
             'Per-source attention mechanism template.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    if not p.context_dim:
      p.context_dim = p.source_dim

    source_atten_tpls = []
    for idx in range(p.num_source):
      key = f'source_{idx}'
      src_atten = p.atten.Copy()
      src_atten.name = f'atten_{key}'
      src_atten.source_dim = p.source_dim
      src_atten.query_dim = p.query_dim
      src_atten.packed_input = True
      source_atten_tpls.append((key, src_atten))

    msa_p = attention.MultiSourceAttention.Params().Set(
        name='multi_source_atten',
        source_dim=p.source_dim,
        query_dim=p.query_dim,
        source_atten_tpls=source_atten_tpls,
        primary_source_key=f'source_{p.primary_source_index}')
    self.CreateChild('atten', msa_p)

  def PrepareExternalInputs(self, theta, external_inputs):
    """Packs source encoder outputs and paddings before decoding."""
    packed = external_inputs.DeepCopy()
    packed.packed_src = self.atten.InitForSourcePacked(
        theta.atten,
        external_inputs.src,
        external_inputs.src,
        external_inputs.padding)
    return packed

  def ZeroState(self, theta, prepared_inputs, batch_size):
    """Initializes per-source attention states and initial context vector."""
    seq_lengths = py_utils.NestedMap()
    for key, src_tensor in prepared_inputs.src.items():
      seq_lengths[key] = py_utils.GetShape(src_tensor, 3)[0]

    init_atten_state = self.atten.ZeroAttentionState(seq_lengths, batch_size)
    zero_query = tf.zeros(
        [batch_size, self.params.query_dim],
        dtype=py_utils.FPropDtype(self.params))
    init_ctx, _, next_atten_state = self.atten.ComputeContextVectorWithSource(
        theta.atten,
        prepared_inputs.packed_src,
        zero_query,
        attention_state=init_atten_state)
    return py_utils.NestedMap(
        atten_context=init_ctx, atten_state=next_atten_state)

  def FProp(self, theta, prepared_inputs, step_inputs, padding, state0):
    """Computes the combined multi-source context vector for one decode step."""
    query = tf.concat(step_inputs.inputs, axis=-1)
    ctx, probs, next_atten_state = self.atten.ComputeContextVectorWithSource(
        theta.atten,
        prepared_inputs.packed_src,
        query,
        attention_state=state0.atten_state)
    probs = py_utils.ApplyPadding(padding, probs)
    output = py_utils.NestedMap(context=ctx, probs=probs)
    state1 = py_utils.NestedMap(
        atten_context=ctx, atten_state=next_atten_state)
    return output, state1
