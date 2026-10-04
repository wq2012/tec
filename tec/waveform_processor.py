"""Log-Mel spectrogram extraction and Griffin-Lim waveform reconstruction."""

from lingvo import compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils
import numpy as np


def compute_fft_size(frame_length_ms: float, sample_rate_hz: float) -> int:
  """Returns the smallest power-of-two FFT size covering `frame_length_ms`."""
  frame_samples = frame_length_ms * sample_rate_hz / 1000.0
  return int(2 ** np.ceil(np.log2(max(1.0, frame_samples))))


class WaveformProcessor(base_layer.BaseLayer):
  """Extracts log-Mel filterbank features and reconstructs waveforms.

  Implements the acoustic frontend described in Section 2.1 and Table 1 of the
  Textual Echo Cancellation paper (https://arxiv.org/pdf/2008.06006):
  - 24 kHz sampling rate
  - 50 ms Hann window (1,200 samples), 12.5 ms frame shift (300 samples)
  - 2,048-point FFT and 128-bin log-Mel filterbank (20 Hz to 12,000 Hz)
  - Iterative Griffin-Lim phase reconstruction for waveform synthesis
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'waveform_processor'
    p.Define('sampling_rate', 24000.0, 'Audio sample rate in Hz.')
    p.Define('frame_length', 50.0, 'Analysis window length in milliseconds.')
    p.Define('frame_step', 12.5, 'Frame hop size in milliseconds.')
    p.Define('fft_size', None, 'FFT size in samples (defaults to power of 2).')
    p.Define('num_mel_bins', 128, 'Number of Mel frequency bins.')
    p.Define('mel_lower_edge_hertz', 20.0, 'Lowest Mel filterbank frequency.')
    p.Define('mel_upper_edge_hertz', 12000.0,
             'Highest Mel filterbank frequency.')
    p.Define('log_dynamic_range_compression', True,
             'Whether to apply natural-log compression to Mel energies.')
    p.Define('magnitude_floor', 1e-3,
             'Minimum magnitude clamp prior to logarithmic compression.')
    p.Define('num_griffin_lim_iters', 100,
             'Number of Griffin-Lim phase estimation iterations.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self._frame_length_samples = int(
        round(p.frame_length * p.sampling_rate / 1000.0))
    self._frame_step_samples = int(
        round(p.frame_step * p.sampling_rate / 1000.0))
    if not p.fft_size:
      p.fft_size = compute_fft_size(p.frame_length, p.sampling_rate)
    if p.fft_size < self._frame_length_samples:
      raise ValueError('fft_size must be >= frame_length in samples.')
    if p.num_griffin_lim_iters < 1:
      raise ValueError('num_griffin_lim_iters must be >= 1.')

  @property
  def num_fft_bins(self) -> int:
    return self.params.fft_size // 2 + 1

  def SamplesToFrames(self, num_samples: int) -> int:
    """Computes the number of STFT frames produced from `num_samples`."""
    if num_samples < self._frame_length_samples:
      return 0
    return (num_samples - self._frame_length_samples) // (
        self._frame_step_samples) + 1

  def FramesToSamples(self, num_frames: int) -> int:
    """Computes the number of waveform samples reconstructed from frames."""
    if num_frames <= 0:
      return 0
    return (num_frames - 1) * self._frame_step_samples + (
        self._frame_length_samples)

  def _mel_weight_matrix(self):
    p = self.params
    weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=p.num_mel_bins,
        num_spectrogram_bins=self.num_fft_bins,
        sample_rate=p.sampling_rate,
        lower_edge_hertz=p.mel_lower_edge_hertz,
        upper_edge_hertz=p.mel_upper_edge_hertz,
        dtype=p.dtype)
    col_sums = tf.reduce_sum(weights, axis=0, keepdims=True)
    return weights / tf.maximum(col_sums, 1e-12)

  def WaveformsToSpectrograms(self, waveforms, waveform_paddings=None):
    """Converts batched time-domain waveforms into log-Mel spectrograms.

    Args:
      waveforms: Float tensor of shape [batch, num_samples].
      waveform_paddings: Optional float tensor of shape [batch, num_samples]
        with 0.0 for valid samples and 1.0 for padded samples.

    Returns:
      NestedMap with `spectrograms` ([batch, frames, num_mel_bins]),
      `raw_spectrograms` ([batch, frames, num_fft_bins]), and `paddings`
      ([batch, frames]).
    """
    p = self.params
    with tf.name_scope(p.name):
      if waveform_paddings is None:
        waveform_paddings = tf.zeros_like(waveforms)

      stft = tf.signal.stft(
          waveforms,
          frame_length=self._frame_length_samples,
          frame_step=self._frame_step_samples,
          fft_length=p.fft_size,
          window_fn=tf.signal.hann_window,
          pad_end=False)
      linear_mag = tf.abs(stft)

      mel_matrix = self._mel_weight_matrix()
      mel_spec = tf.tensordot(linear_mag, mel_matrix, axes=[[2], [0]])

      if p.log_dynamic_range_compression:
        floor = float(p.magnitude_floor)
        mel_spec = tf.math.log(tf.maximum(floor, mel_spec))
        raw_spec = tf.math.log(tf.maximum(floor, linear_mag))
      else:
        raw_spec = linear_mag

      framed_pad = tf.signal.frame(
          waveform_paddings,
          frame_length=self._frame_length_samples,
          frame_step=self._frame_step_samples,
          pad_end=False)
      frame_paddings = tf.reduce_max(framed_pad, axis=-1)

      valid_mask = 1.0 - tf.expand_dims(frame_paddings, axis=-1)
      mel_spec *= valid_mask
      raw_spec *= valid_mask

      return py_utils.NestedMap(
          spectrograms=mel_spec,
          raw_spectrograms=raw_spec,
          paddings=frame_paddings)

  def SpectrogramsToWaveforms(self,
                              mel_spectrograms,
                              spectrogram_paddings=None,
                              reference_waveforms=None):
    """Synthesizes time-domain waveforms from log-Mel spectrograms.

    Args:
      mel_spectrograms: Float tensor of shape [batch, frames, num_mel_bins].
      spectrogram_paddings: Optional float tensor of shape [batch, frames].
      reference_waveforms: Optional microphone mixture waveforms of shape
        [batch, num_samples] used for phase-preserving Mel-band gain synthesis.

    Returns:
      Tuple of (waveforms, waveform_paddings) of shape [batch, num_samples].
    """
    p = self.params
    with tf.name_scope(p.name):
      if spectrogram_paddings is None:
        spectrogram_paddings = tf.zeros(
            tf.shape(mel_spectrograms)[:2], dtype=mel_spectrograms.dtype)

      mel_mag = (
          tf.exp(mel_spectrograms)
          if p.log_dynamic_range_compression else mel_spectrograms)
      valid_frame_mask = 1.0 - tf.expand_dims(spectrogram_paddings, axis=-1)
      mel_mag *= valid_frame_mask

      if reference_waveforms is not None:
        ref_stft = tf.signal.stft(
            reference_waveforms,
            frame_length=self._frame_length_samples,
            frame_step=self._frame_step_samples,
            fft_length=p.fft_size,
            window_fn=tf.signal.hann_window,
            pad_end=False)
        num_pred_frames = tf.shape(mel_mag)[1]
        num_ref_frames = tf.shape(ref_stft)[1]
        common_frames = tf.minimum(num_pred_frames, num_ref_frames)
        ref_stft_slice = ref_stft[:, :common_frames, :]
        ref_mag = tf.abs(ref_stft_slice)
        mel_matrix = self._mel_weight_matrix()
        ref_mel_mag = tf.tensordot(ref_mag, mel_matrix, axes=[[2], [0]])
        floor = float(p.magnitude_floor)
        mel_gain = tf.clip_by_value(
            mel_mag[:, :common_frames, :] / tf.maximum(ref_mel_mag, floor),
            0.0,
            1.2)
        raw_weights = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=p.num_mel_bins,
            num_spectrogram_bins=self.num_fft_bins,
            sample_rate=p.sampling_rate,
            lower_edge_hertz=p.mel_lower_edge_hertz,
            upper_edge_hertz=p.mel_upper_edge_hertz,
            dtype=p.dtype)
        row_sums = tf.reduce_sum(raw_weights, axis=1, keepdims=True)
        mel_to_lin = tf.transpose(raw_weights / tf.maximum(row_sums, 1e-12))
        linear_gain = tf.tensordot(mel_gain, mel_to_lin, axes=[[2], [0]])
        enhanced_stft = ref_stft_slice * tf.cast(linear_gain, tf.complex64)
        rem_frames = num_pred_frames - common_frames
        enhanced_stft = tf.pad(
            enhanced_stft, [[0, 0], [0, rem_frames], [0, 0]])
        inv_window_fn = tf.signal.inverse_stft_window_fn(
            self._frame_step_samples, forward_window_fn=tf.signal.hann_window)
        waveforms = tf.signal.inverse_stft(
            enhanced_stft,
            frame_length=self._frame_length_samples,
            frame_step=self._frame_step_samples,
            fft_length=p.fft_size,
            window_fn=inv_window_fn)
      else:
        mel_pinv = tf.linalg.pinv(self._mel_weight_matrix())
        linear_mag = tf.tensordot(mel_mag, mel_pinv, axes=[[2], [0]])
        linear_mag = tf.maximum(float(p.magnitude_floor), linear_mag)
        linear_mag *= valid_frame_mask
        waveforms = self._griffin_lim_reconstruct(linear_mag)

      batch_size = tf.shape(spectrogram_paddings)[0]
      num_frames = tf.shape(spectrogram_paddings)[1]
      expanded_pad = tf.tile(
          tf.expand_dims(spectrogram_paddings, axis=-1),
          [1, 1, self._frame_length_samples])
      overlap_pad = tf.signal.overlap_and_add(
          expanded_pad, self._frame_step_samples)
      total_samples = (
          (num_frames - 1) * self._frame_step_samples +
          self._frame_length_samples)
      overlap_pad = tf.reshape(overlap_pad, [batch_size, total_samples])
      waveform_paddings = tf.cast(overlap_pad > 0.0, p.dtype)

      waveforms *= (1.0 - waveform_paddings)
      return waveforms, waveform_paddings

  def _griffin_lim_reconstruct(self, linear_magnitude):
    """Runs iterative Griffin-Lim phase estimation on linear magnitudes."""
    p = self.params
    mag_c64 = tf.cast(linear_magnitude, tf.complex64)
    num_frames = tf.shape(mag_c64)[1]
    inv_window_fn = tf.signal.inverse_stft_window_fn(
        self._frame_step_samples, forward_window_fn=tf.signal.hann_window)

    def _step(idx, cur_stft):
      wav = tf.signal.inverse_stft(
          cur_stft,
          frame_length=self._frame_length_samples,
          frame_step=self._frame_step_samples,
          fft_length=p.fft_size,
          window_fn=inv_window_fn)
      est_stft = tf.signal.stft(
          wav,
          frame_length=self._frame_length_samples,
          frame_step=self._frame_step_samples,
          fft_length=p.fft_size,
          window_fn=tf.signal.hann_window,
          pad_end=False)[:, :num_frames, :]
      unit_phase = est_stft / tf.cast(
          tf.maximum(1e-8, tf.abs(est_stft)), tf.complex64)
      return idx + 1, mag_c64 * unit_phase

    _, final_stft = tf.while_loop(
        lambda idx, _: idx < p.num_griffin_lim_iters,
        _step,
        loop_vars=[tf.constant(0), mag_c64],
        back_prop=False)

    return tf.signal.inverse_stft(
        final_stft,
        frame_length=self._frame_length_samples,
        frame_step=self._frame_step_samples,
        fft_length=p.fft_size,
        window_fn=inv_window_fn)
