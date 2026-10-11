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
        encoder_speech.SpeechEncoderV1.Params().Set(
            input_residual_weight=-0.45, active_energy_only=True),
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
      filter_length: int = 128,
      step_size: float = 0.00018,
      eps: float = 1e-3,
      use_postfilter: bool = False,
  ):
    if filter_length < 1:
      raise ValueError('filter_length must be >= 1.')
    if not 0.0 < step_size < 2.0:
      raise ValueError('step_size must be in (0, 2).')
    self.filter_length = filter_length
    self.step_size = step_size
    self.eps = eps
    self.use_postfilter = use_postfilter

  @staticmethod
  def _apply_residual_postfilter(
      error_sig: np.ndarray,
      reference_sig: np.ndarray,
  ) -> np.ndarray:
    """Applies a causal STFT Wiener post-filter to suppress residual echo."""
    n_samples = len(error_sig)
    n_fft = 1024
    hop = 256
    if n_samples < n_fft:
      return error_sig

    win = np.hanning(n_fft).astype(np.float64)
    pad_len = n_fft
    err_pad = np.pad(error_sig, (0, pad_len))
    ref_pad = np.pad(reference_sig, (0, pad_len))
    n_frames = 1 + (len(err_pad) - n_fft) // hop

    err_frames = np.stack(
        [err_pad[i * hop:i * hop + n_fft] * win for i in range(n_frames)],
        axis=0)
    ref_frames = np.stack(
        [ref_pad[i * hop:i * hop + n_fft] * win for i in range(n_frames)],
        axis=0)

    err_spec = np.fft.rfft(err_frames, axis=-1)
    ref_spec = np.fft.rfft(ref_frames, axis=-1)
    err_mag = np.abs(err_spec)
    ref_mag = np.abs(ref_spec)

    # Model multi-frame room reverberation envelope from causal reference lags.
    ref_rev = 0.55 * ref_mag.copy()
    if n_frames > 1:
      ref_rev[1:] += 0.28 * ref_mag[:-1]
    if n_frames > 2:
      ref_rev[2:] += 0.17 * ref_mag[:-2]

    scale = np.sum(err_mag * ref_rev, axis=0, keepdims=True) / (
        np.sum(ref_rev * ref_rev, axis=0, keepdims=True) + 1e-8)
    scale = np.clip(scale, 0.0, 1.5)
    echo_mag_est = scale * ref_rev
    gain = np.maximum(
        0.18,
        np.maximum(0.0, err_mag - 0.55 * echo_mag_est) / (err_mag + 1e-8))
    enhanced_frames = np.fft.irfft(err_spec * gain, n=n_fft, axis=-1) * win

    out = np.zeros(len(err_pad), dtype=np.float64)
    win_norm = np.zeros(len(err_pad), dtype=np.float64)
    win_sq = win * win
    for i in range(n_frames):
      start = i * hop
      out[start:start + n_fft] += enhanced_frames[i]
      win_norm[start:start + n_fft] += win_sq
    out = out / np.maximum(1e-8, win_norm)
    return out[:n_samples]

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
    weights = np.zeros(taps, dtype=np.float64)
    x_rev = np.pad(x, (taps - 1, 0))[::-1]
    total_len = len(x_rev)
    x_norm_sq = np.convolve(
        x * x, np.ones(taps, dtype=np.float64), mode='full')[:num_samples]

    error_signal = np.zeros(num_samples, dtype=np.float64)
    echo_estimate = np.zeros(num_samples, dtype=np.float64)
    mu = self.step_size
    eps = self.eps
    dt_reg = 0.5 * taps if taps >= 128 else 0.0

    for n in range(num_samples):
      start = total_len - taps - n
      x_vec = x_rev[start:start + taps]
      y_hat = float(np.dot(weights, x_vec))
      err = d[n] - y_hat
      norm = float(x_norm_sq[n]) + eps + dt_reg * (err * err)
      weights += (mu * err / norm) * x_vec
      echo_estimate[n] = y_hat
      error_signal[n] = err

    if self.use_postfilter and taps >= 128 and num_samples >= 1024:
      error_signal = self._apply_residual_postfilter(error_signal, x)
      echo_estimate = d - error_signal

    return (
        error_signal.astype(np.float32),
        echo_estimate.astype(np.float32),
        weights.astype(np.float32),
    )
