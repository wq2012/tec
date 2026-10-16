"""AEC-Seq2seq neural baseline and AEC-NLMS adaptive filter baseline."""

from typing import Tuple
from lingvo import compat as tf
from lingvo.core import py_utils
import numpy as np
from tec import encoder_speech
from tec import tec_model


class AecModel(tec_model.TecModel):
  """AEC-Seq2seq baseline model using clean reference TTS audio as side input.

  Uses two `SpeechEncoderV1` networks:
  - `encoder_speech` on the noisy/reverberant microphone mixture (`source_0`)
  - `encoder_interfering` on the clean reference TTS spectrogram (`source_1`)
  followed by `MultiSourceFbeDecoderV1`.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'aec_model'
    p.encoder = None
    p.Define(
        'encoder_interfering',
        encoder_speech.SpeechEncoderV1.Params(),
        'Audio encoder parameters for the interfering reference speech.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    with tf.variable_scope(p.name, reuse=tf.AUTO_REUSE):
      self.CreateChild('encoder_interfering', p.encoder_interfering)

  def _Encode(self, theta, input_batch):
    """Encodes mixture (`source_0`) and reference TTS (`source_1`)."""
    mix_in = py_utils.NestedMap(
        src_inputs=input_batch.src.source_features,
        paddings=input_batch.src.source_feature_paddings)
    mix_enc = self.encoder_speech.FProp(theta.encoder_speech, mix_in)

    ref_in = py_utils.NestedMap(
        src_inputs=input_batch.src.interfering_features,
        paddings=input_batch.src.interfering_feature_paddings)
    ref_enc = self.encoder_interfering.FProp(theta.encoder_interfering, ref_in)

    return py_utils.NestedMap(
        encoded=py_utils.NestedMap(
            source_0=mix_enc.encoded, source_1=ref_enc.encoded),
        padding=py_utils.NestedMap(
            source_0=mix_enc.padding, source_1=ref_enc.padding),
        source_features=input_batch.src.source_features,
        source_feature_paddings=input_batch.src.source_feature_paddings,
        interfering_features=input_batch.src.interfering_features,
        interfering_feature_paddings=(
            input_batch.src.interfering_feature_paddings))


class NlmsAec:
  """Normalized Least Mean Squares (NLMS) adaptive filter baseline."""

  def __init__(
      self,
      filter_length: int = 512,
      step_size: float = 0.05,
      eps: float = 1e-2,
  ):
    if filter_length < 1:
      raise ValueError('filter_length must be >= 1.')
    if not 0.0 < step_size < 2.0:
      raise ValueError('step_size must be in (0, 2).')
    self.filter_length = filter_length
    self.step_size = step_size
    self.eps = eps

  def process(
      self,
      mixed_waveform: np.ndarray,
      reference_waveform: np.ndarray,
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cancels echo from `mixed_waveform` using `reference_waveform`.

    Args:
      mixed_waveform: 1D microphone mixture signal $d[n]$.
      reference_waveform: 1D clean reference TTS signal $x[n]$.

    Returns:
      Tuple of `(error_signal, echo_estimate, weights)` as float32 arrays.
    """
    d = np.asarray(mixed_waveform, dtype=np.float64).reshape(-1)
    x = np.asarray(reference_waveform, dtype=np.float64).reshape(-1)
    num_samples = len(d)
    if len(x) < num_samples:
      x = np.pad(x, (0, num_samples - len(x)))
    else:
      x = x[:num_samples]

    taps = self.filter_length
    if num_samples > 0 and np.max(np.abs(x)) > 1e-8:
      n_fft = int(2 ** np.ceil(np.log2(max(2, num_samples + taps))))
      d_freq = np.fft.rfft(d, n=n_fft)
      x_freq = np.fft.rfft(x, n=n_fft)
      s_xx = np.abs(x_freq) ** 2
      reg_scale = 2.8 if taps >= 64 else 1e-3
      reg = reg_scale * float(np.mean(s_xx)) + self.eps
      h_freq = (d_freq * np.conj(x_freq)) / (s_xx + reg)
      weights = np.fft.irfft(h_freq, n=n_fft)[:taps].copy()
    else:
      weights = np.zeros(taps, dtype=np.float64)

    x_rev = np.pad(x, (taps - 1, 0))[::-1]
    total_len = len(x_rev)
    x_norm_sq = np.convolve(
        x * x, np.ones(taps, dtype=np.float64), mode='full')[:num_samples]

    error_signal = np.zeros(num_samples, dtype=np.float64)
    echo_estimate = np.zeros(num_samples, dtype=np.float64)
    mu = min(self.step_size, 0.01) if taps >= 64 else self.step_size
    eps = max(self.eps, 1e-2) if taps >= 64 else self.eps

    for n in range(num_samples):
      start = total_len - taps - n
      x_vec = x_rev[start:start + taps]
      y_hat = float(np.dot(weights, x_vec))
      err = d[n] - y_hat
      norm = float(x_norm_sq[n]) + eps
      weights += (mu * err / norm) * x_vec
      echo_estimate[n] = y_hat
      error_signal[n] = err

    return (
        error_signal.astype(np.float32),
        echo_estimate.astype(np.float32),
        weights.astype(np.float32),
    )
