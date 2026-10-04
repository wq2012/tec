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
    p.Define('encoder_interfering', encoder_speech.SpeechEncoderV1.Params(),
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
            source_0=mix_enc.padding, source_1=ref_enc.padding))


class NlmsAec:
  """Normalized Least Mean Squares (NLMS) adaptive filter baseline."""

  def __init__(
      self,
      filter_length: int = 1024,
      step_size: float = 0.2,
      eps: float = 1e-6,
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
    """Cancels linear echo from `mixed_waveform` using `reference_waveform`.

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
    # Reverse padded reference so each window [n : n + taps] is
    # [x[n], x[n - 1], ..., x[n - taps + 1]] without per-step reversal copies.
    x_rev = np.pad(x, (taps - 1, 0))[::-1]
    total_len = len(x_rev)

    error_signal = np.zeros(num_samples, dtype=np.float64)
    echo_estimate = np.zeros(num_samples, dtype=np.float64)
    mu = self.step_size
    eps = self.eps

    for n in range(num_samples):
      start = total_len - taps - n
      x_vec = x_rev[start:start + taps]
      y_hat = float(np.dot(weights, x_vec))
      err = d[n] - y_hat
      norm = float(np.dot(x_vec, x_vec)) + eps
      weights += (mu * err / norm) * x_vec
      echo_estimate[n] = y_hat
      error_signal[n] = err

    return (
        error_signal.astype(np.float32),
        echo_estimate.astype(np.float32),
        weights.astype(np.float32),
    )
