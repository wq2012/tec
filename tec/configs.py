"""Lingvo model configurations for Textual Echo Cancellation."""

from lingvo import model_registry
from lingvo.core import attention
from lingvo.core import base_model_params
from lingvo.core import optimizer
from lingvo.core import py_utils
from lingvo.core import rnn_cell
from lingvo.core import schedule as lr_schedule
from tec import aec_model
from tec import decoder as tec_decoder
from tec import encoder_speech
from tec import encoder_text
from tec import input_generator
from tec import tec_model
from tec import waveform_processor


def configure_waveform_processor(
    sampling_rate: float = 24000.0,
    num_mel_bins: int = 128,
):
  """Returns `WaveformProcessor` parameters matching Table 1 of the paper."""
  p = waveform_processor.WaveformProcessor.Params()
  p.sampling_rate = sampling_rate
  p.num_mel_bins = num_mel_bins
  p.frame_length = 50.0
  p.frame_step = 12.5
  p.mel_lower_edge_hertz = 20.0
  p.mel_upper_edge_hertz = sampling_rate / 2.0
  p.log_dynamic_range_compression = True
  p.magnitude_floor = 1e-3
  p.num_griffin_lim_iters = 100
  return p


def configure_speech_encoder(
    num_mel_bins: int = 128,
    lstm_cell_size: int = 256,
    num_lstm_layers: int = 3,
    num_conv_lstm_layers: int = 1,
):
  """Returns `SpeechEncoderV1` parameters matching Table 1 of the paper."""
  p = encoder_speech.SpeechEncoderV1.Params()
  p.use_specaugment = False
  p.input_shape = [None, None, num_mel_bins, 1]
  p.conv_filter_shapes = [(3, 3, 1, 32), (3, 3, 32, 32)]
  p.conv_filter_strides = [(2, 2), (2, 2)]
  p.num_conv_lstm_layers = num_conv_lstm_layers
  p.num_lstm_layers = num_lstm_layers
  p.lstm_cell_size = lstm_cell_size
  p.project_lstm_output = False
  p.pad_steps = 0
  return p


def configure_text_encoder(
    vocab_size: int = 96,
    embedding_dim: int = 512,
    lstm_cell_size: int = 256,
):
  """Returns `TtsEncoderV2` parameters matching Table 1 of the paper."""
  p = encoder_text.TtsEncoderV2.Params()
  p.emb.vocab_size = vocab_size
  p.emb.embedding_dim = embedding_dim
  p.emb.scale_sqrt_depth = False
  p.filter_shapes = [(5, 1, 512, 512), (5, 1, 512, 512), (5, 1, 512, 512)]
  p.filter_strides = [(1, 1), (1, 1), (1, 1)]
  p.dropout_prob = 0.5
  p.zoneout_prob = 0.1
  p.lstm_cell_size = lstm_cell_size
  return p


def configure_multi_source_decoder(
    feature_dims: int = 128,
    source_dim: int = 512,
    use_gmm_attention: bool = True,
):
  """Returns `MultiSourceFbeDecoderV1` parameters matching Table 1."""
  p = tec_decoder.MultiSourceFbeDecoderV1.Params()
  p.feature_dims = feature_dims
  p.source_dim = source_dim
  p.reduction_factor = 4
  p.target_eos_offset_frames = 1
  p.eos_loss_weight = 1.0
  p.l1_loss_weight = 1.0
  p.l2_loss_weight = 1.0
  p.decode_max_output_frames = 800
  p.step.rnn_layers = 2
  p.step.rnn_cell_dim = 256
  p.step.rnn_cell_tpl = rnn_cell.LSTMCellSimple.Params().Set(
      deterministic=True,
      zo_prob=0.1,
      params_init=py_utils.WeightInit.Uniform(0.1))
  p.step.target_pre_net.hidden_layer_dims = [256, 256]
  p.step.target_pre_net.dropout.keep_prob = 0.5
  p.step.target_pre_net.dropout.dropout_at_eval = False
  p.step.eos_prob_threshold = 0.5
  if use_gmm_attention:
    p.step.attention = attention.GmmMonotonicAttention.Params().Set(
        hidden_dim=128, num_mixtures=5)
  else:
    p.step.attention = attention.AdditiveAttention.Params().Set(hidden_dim=128)
  return p


class TecBaseConfig(base_model_params.SingleTaskModelParams):
  """Base configuration for Textual Echo Cancellation on 24 kHz audio."""

  NUM_MEL_BINS = 128
  SAMPLING_RATE = 24000.0
  VOCAB_SIZE = 96
  TRAIN_FILE_PATTERN = ''
  DEV_FILE_PATTERN = ''
  TEST_FILE_PATTERN = ''

  def _InputParams(self, file_pattern: str, is_eval: bool = False):
    p = input_generator.TecInputGenerator.Params()
    p.file_pattern = file_pattern
    p.batch_size = 4 if is_eval else 16
    p.vocab_size = self.VOCAB_SIZE
    p.waveform_processor = configure_waveform_processor(
        self.SAMPLING_RATE, self.NUM_MEL_BINS)
    p.shuffle = not is_eval
    p.repeat = not is_eval
    p.use_synthetic_data = not bool(file_pattern)
    return p

  def Train(self):
    return self._InputParams(self.TRAIN_FILE_PATTERN, is_eval=False)

  def Dev(self):
    return self._InputParams(self.DEV_FILE_PATTERN, is_eval=True)

  def Test(self):
    return self._InputParams(self.TEST_FILE_PATTERN, is_eval=True)

  def Task(self):
    p = tec_model.TecModel.Params()
    p.name = 'tec'
    p.waveform_processor = configure_waveform_processor(
        self.SAMPLING_RATE, self.NUM_MEL_BINS)
    p.encoder_speech = configure_speech_encoder(
        num_mel_bins=self.NUM_MEL_BINS,
        lstm_cell_size=256,
        num_lstm_layers=3,
        num_conv_lstm_layers=0)
    p.encoder = configure_text_encoder(
        vocab_size=self.VOCAB_SIZE,
        embedding_dim=512,
        lstm_cell_size=256)
    p.decoder = configure_multi_source_decoder(
        feature_dims=self.NUM_MEL_BINS,
        source_dim=512,
        use_gmm_attention=True)

    tp = p.train
    tp.learning_rate = 1e-3
    tp.lr_schedule = lr_schedule.PiecewiseConstantSchedule.Params().Set(
        boundaries=[20000, 40000, 60000],
        values=[1.0, 0.5, 0.25, 0.125])
    tp.optimizer = optimizer.Adam.Params().Set(
        beta1=0.9, beta2=0.999, epsilon=1e-6)
    tp.clip_gradient_norm_to_value = 1.0
    tp.ema_decay = 0.9999
    return p


@model_registry.RegisterSingleTaskModel
class TecSingleInterfering(TecBaseConfig):
  """TEC config for single-speaker TTS interference (LibriTTS + LJSpeech)."""
  pass


@model_registry.RegisterSingleTaskModel
class TecMultiInterfering(TecBaseConfig):
  """TEC configuration for multi-speaker TTS interference (LibriTTS + VCTK)."""
  pass


@model_registry.RegisterSingleTaskModel
class NoSideInputSingleInterfering(TecBaseConfig):
  """Vanilla-Seq2seq baseline (no side input) on LibriTTS + LJSpeech."""

  def Task(self):
    p = super().Task()
    p.name = 'no_side_input'
    p.encoder = None
    p.encoder_speech.num_conv_lstm_layers = 0
    dec_p = tec_decoder.FbeDecoderV1.Params()
    dec_p.feature_dims = p.decoder.feature_dims
    dec_p.source_dim = p.decoder.source_dim
    dec_p.reduction_factor = p.decoder.reduction_factor
    dec_p.target_eos_offset_frames = p.decoder.target_eos_offset_frames
    dec_p.decode_max_output_frames = p.decoder.decode_max_output_frames
    dec_p.step.rnn_layers = p.decoder.step.rnn_layers
    dec_p.step.rnn_cell_dim = p.decoder.step.rnn_cell_dim
    dec_p.step.attention = p.decoder.step.attention.Copy()
    p.decoder = dec_p
    return p


@model_registry.RegisterSingleTaskModel
class NoSideInputMultiInterfering(NoSideInputSingleInterfering):
  """Vanilla-Seq2seq baseline (no side input) on LibriTTS + VCTK."""
  pass


@model_registry.RegisterSingleTaskModel
class AecSingleInterfering(TecBaseConfig):
  """AEC-Seq2seq baseline (audio side input) on LibriTTS + LJSpeech."""

  def Task(self):
    base_p = super().Task()
    p = aec_model.AecModel.Params()
    p.name = 'aec_seq2seq'
    p.waveform_processor = base_p.waveform_processor
    p.encoder = None
    p.encoder_speech = base_p.encoder_speech.Copy()
    p.encoder_interfering = base_p.encoder_speech.Copy().Set(
        name='interfering_speech_encoder')
    p.decoder = base_p.decoder
    p.train = base_p.train
    return p


@model_registry.RegisterSingleTaskModel
class AecMultiInterfering(AecSingleInterfering):
  """AEC-Seq2seq baseline (audio side input) on LibriTTS + VCTK."""
  pass
