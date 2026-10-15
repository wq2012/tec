"""Evaluation metrics for Textual Echo Cancellation (MCD, DTW, WER, FLOPS)."""

import re
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
from scipy import fftpack
from scipy import stats

_PUNCTUATION_CHARS = r'!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~'
_PUNCTUATION_RE = re.compile(f'[{_PUNCTUATION_CHARS}]')
_MCD_DB_SCALE = (10.0 * np.sqrt(2.0)) / np.log(10.0)


def compute_mfcc_from_log_mel(
    log_mel_spectrogram: np.ndarray,
    num_mfccs: int = 14,
) -> np.ndarray:
  """Computes MFCCs from a log-Mel spectrogram via type-II DCT.

  Args:
    log_mel_spectrogram: 2D array of shape `[frames, num_mel_bins]`.
    num_mfccs: Number of DCT coefficients to return (including the 0th energy
      coefficient `c_0`). Defaults to 14 so that slicing `[:, 1:14]` yields all
      13 cepstral coefficients `d = 1..13` required by Equation 12 of the paper.

  Returns:
    2D float64 array of shape `[frames, num_mfccs]`.
  """
  mel = np.asarray(log_mel_spectrogram, dtype=np.float64)
  if mel.ndim != 2:
    raise ValueError(
        f'Expected 2D log_mel_spectrogram [frames, bins], got {mel.shape}')
  num_bins = mel.shape[1]
  dct_all = fftpack.dct(mel, type=2, axis=-1, norm=None)
  scale = 1.0 / float(num_bins)
  return (dct_all * scale)[:, :num_mfccs]


def compute_dtw_distance(
    seq_ref: np.ndarray,
    seq_pred: np.ndarray,
) -> Tuple[float, int]:
  """Computes Dynamic Time Warping (DTW) Euclidean distance between matrices."""
  ref = np.asarray(seq_ref, dtype=np.float64)
  pred = np.asarray(seq_pred, dtype=np.float64)
  n, m = len(ref), len(pred)
  if n == 0 or m == 0:
    raise ValueError(
        'Input sequences to compute_dtw_distance must be non-empty')

  diff = ref[:, None, :] - pred[None, :, :]
  dist_matrix = np.linalg.norm(diff, axis=-1)

  dist_rows = dist_matrix.tolist()
  inf = float('inf')
  cost = [[inf] * (m + 1) for _ in range(n + 1)]
  steps = [[0] * (m + 1) for _ in range(n + 1)]
  cost[0][0] = 0.0

  for i in range(1, n + 1):
    d_row = dist_rows[i - 1]
    c_prev = cost[i - 1]
    c_curr = cost[i]
    s_prev = steps[i - 1]
    s_curr = steps[i]
    for j in range(1, m + 1):
      c_diag = c_prev[j - 1]
      c_up = c_prev[j]
      c_left = c_curr[j - 1]
      if c_diag <= c_up and c_diag <= c_left:
        c_curr[j] = d_row[j - 1] + c_diag
        s_curr[j] = s_prev[j - 1] + 1
      elif c_up <= c_left:
        c_curr[j] = d_row[j - 1] + c_up
        s_curr[j] = s_prev[j] + 1
      else:
        c_curr[j] = d_row[j - 1] + c_left
        s_curr[j] = s_curr[j - 1] + 1

  return float(cost[n][m]), int(steps[n][m])


def compute_mcd(
    log_mel_ref: np.ndarray,
    log_mel_pred: np.ndarray,
    num_mfccs: int = 13,
    use_dtw: bool = True,
) -> float:
  """Computes Mel Cepstral Distortion (MCD) in dB between two spectrograms.

  Implements Section 3.3.2 (Equation 12) of the paper:
    MCD = (10 / ln 10) * (1 / T)
          * sum_t sqrt(2 * sum_{d=1}^{13} (z^_td - z_td)^2)
  excluding the 0th cepstral coefficient `c_0` and using `num_mfccs = 13`
  cepstral coefficients (`d = 1..13`) with DTW time alignment.
  """
  mfcc_ref = compute_mfcc_from_log_mel(
      log_mel_ref, num_mfccs=num_mfccs + 1)[:, 1:num_mfccs + 1]
  mfcc_pred = compute_mfcc_from_log_mel(
      log_mel_pred, num_mfccs=num_mfccs + 1)[:, 1:num_mfccs + 1]

  if use_dtw:
    total_dist, _ = compute_dtw_distance(mfcc_ref, mfcc_pred)
    mean_dist = float(total_dist / max(len(mfcc_ref), len(mfcc_pred)))
  else:
    min_len = min(len(mfcc_ref), len(mfcc_pred))
    frame_dists = np.linalg.norm(
        mfcc_ref[:min_len] - mfcc_pred[:min_len], axis=-1)
    mean_dist = float(np.mean(frame_dists))
  return float(_MCD_DB_SCALE * mean_dist)


def normalize_transcript(text: str) -> str:
  """Strips punctuation and normalizes whitespace and case for WER scoring."""
  cleaned = _PUNCTUATION_RE.sub('', text)
  return re.sub(r'\s+', ' ', cleaned).strip().lower()


def compute_wer(
    references: Sequence[str],
    hypotheses: Sequence[str],
    normalize: bool = True,
) -> Dict[str, object]:
  """Computes Word Error Rate (WER) across reference and hypothesis pairs."""
  if len(references) != len(hypotheses):
    raise ValueError(
        f'Length mismatch: {len(references)} references vs '
        f'{len(hypotheses)} hypotheses')

  total_edits = 0
  total_words = 0
  per_utt_edits = []
  per_utt_words = []

  for ref, hyp in zip(references, hypotheses):
    if normalize:
      ref = normalize_transcript(ref)
      hyp = normalize_transcript(hyp)
    ref_words = ref.split()
    hyp_words = hyp.split()
    n, m = len(ref_words), len(hyp_words)
    total_words += n

    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
      for j in range(1, m + 1):
        cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
        dp[i, j] = min(
            dp[i - 1, j] + 1,
            dp[i, j - 1] + 1,
            dp[i - 1, j - 1] + cost)
    edits = int(dp[n, m])
    total_edits += edits
    per_utt_edits.append(edits)
    per_utt_words.append(n)

  wer_pct = (100.0 * total_edits / total_words) if total_words > 0 else 0.0
  return {
      'wer': float(wer_pct),
      'word_errors': int(total_edits),
      'total_words': int(total_words),
      'per_utt_edits': per_utt_edits,
      'per_utt_words': per_utt_words,
  }


def compute_bootstrap_ci(
    values: Sequence[float],
    num_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
  """Computes nonparametric bootstrap confidence interval for a metric.

  When `weights` is None, resamples `values` and computes the sample mean.
  When `weights` is provided (e.g., per-utterance reference word counts for
  corpus WER where `values` are per-utterance word error counts * 100), computes
  the ratio of sums `sum(values[idx]) / sum(weights[idx])` on each resample.
  """
  arr = np.asarray(values, dtype=np.float64)
  n = len(arr)
  if n == 0:
    return {'mean': 0.0, 'std_err': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0}

  rng = np.random.RandomState(seed)
  indices = rng.randint(0, n, size=(num_resamples, n))

  if weights is not None:
    w = np.asarray(weights, dtype=np.float64)
    point_est = float(np.sum(arr) / max(1e-12, np.sum(w)))
    sampled_num = np.sum(arr[indices], axis=1)
    sampled_den = np.maximum(1e-12, np.sum(w[indices], axis=1))
    boot_stats = sampled_num / sampled_den
  else:
    point_est = float(np.mean(arr))
    boot_stats = np.mean(arr[indices], axis=1)

  alpha = (1.0 - confidence) / 2.0
  ci_lower = float(np.percentile(boot_stats, 100.0 * alpha))
  ci_upper = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha)))
  return {
      'mean': point_est,
      'std_err': float(np.std(boot_stats, ddof=1)),
      'ci_lower': ci_lower,
      'ci_upper': ci_upper,
  }


def compute_paired_significance(
    values_a: Sequence[float],
    values_b: Sequence[float],
    num_resamples: int = 10000,
    seed: int = 42,
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
  """Computes paired bootstrap and Wilcoxon signed-rank p-values (A vs B)."""
  a = np.asarray(values_a, dtype=np.float64)
  b = np.asarray(values_b, dtype=np.float64)
  if len(a) != len(b) or len(a) == 0:
    raise ValueError(
        'values_a and values_b must have identical non-zero length')

  n = len(a)
  rng = np.random.RandomState(seed)
  indices = rng.randint(0, n, size=(num_resamples, n))

  if weights is not None:
    w = np.asarray(weights, dtype=np.float64)
    mean_diff = float((np.sum(a) - np.sum(b)) / max(1e-12, np.sum(w)))
    diff_samples = (
        np.sum(a[indices] - b[indices], axis=1) /
        np.maximum(1e-12, np.sum(w[indices], axis=1)))
    per_unit_diff = (a - b) / np.maximum(1.0, w)
  else:
    mean_diff = float(np.mean(a - b))
    diff_samples = np.mean(a[indices] - b[indices], axis=1)
    per_unit_diff = a - b

  p_pos = float(np.mean(diff_samples >= 0.0))
  p_neg = float(np.mean(diff_samples <= 0.0))
  p_bootstrap = float(min(1.0, 2.0 * min(p_pos, p_neg)))

  if np.allclose(per_unit_diff, 0.0):
    p_wilcoxon = 1.0
    t_stat = 0.0
    p_ttest = 1.0
  else:
    try:
      _, p_wilcoxon = stats.wilcoxon(per_unit_diff, zero_method='wilcox')
      p_wilcoxon = float(p_wilcoxon)
    except ValueError:
      p_wilcoxon = p_bootstrap
    try:
      t_res = stats.ttest_rel(a, b) if weights is None else stats.ttest_1samp(
          per_unit_diff, 0.0)
      t_stat = float(t_res.statistic)
      p_ttest = float(t_res.pvalue)
    except ValueError:
      t_stat = 0.0
      p_ttest = p_bootstrap

  return {
      'mean_diff': mean_diff,
      'ci_lower': float(np.percentile(diff_samples, 2.5)),
      'ci_upper': float(np.percentile(diff_samples, 97.5)),
      't_stat': t_stat,
      'p_ttest': p_ttest,
      'p_bootstrap': p_bootstrap,
      'p_wilcoxon': p_wilcoxon,
      'p_value_ttest': p_ttest,
      'p_value_bootstrap': p_bootstrap,
      'p_value_wilcoxon': p_wilcoxon,
  }


def compute_analytical_model_params(
    num_mel_bins: int = 128,
    audio_conv_channels: int = 32,
    audio_lstm_cell_size: int = 256,
    audio_num_lstm_layers: int = 3,
    audio_num_conv_lstm_layers: int = 1,
    text_embedding_dim: int = 512,
    text_conv_channels: int = 512,
    text_num_conv_layers: int = 3,
    text_conv_kernel: int = 5,
    text_lstm_cell_size: int = 256,
    decoder_prenet_dim: int = 256,
    decoder_rnn_layers: int = 2,
    decoder_rnn_cell_dim: int = 256,
    postnet_channels: int = 512,
    postnet_layers: int = 5,
    postnet_kernel: int = 5,
    atten_hidden_dim: int = 128,
    reduction_factor: int = 4,
) -> Dict[str, float]:
  """Computes analytical parameter counts per module (Table 1 & Equation 13)."""
  # 1. Audio Encoder (SpeechEncoderV1): 2 strided 3x3 Conv2D
  # + 1 Bi-CLSTM (1x3 kernel, 256 units/dir) + 3 Bi-LSTM (256 units/dir).
  conv1_params = 3 * 3 * 1 * audio_conv_channels + 3 * audio_conv_channels
  conv2_params = (
      3 * 3 * audio_conv_channels * audio_conv_channels +
      3 * audio_conv_channels)
  bi_clstm_params = audio_num_conv_lstm_layers * (
      2 * 4 * (3 * audio_conv_channels + audio_lstm_cell_size + 1) *
      audio_lstm_cell_size)
  bi_lstm_params = 0
  cur_dim = 2 * audio_lstm_cell_size
  for _ in range(audio_num_lstm_layers):
    bi_lstm_params += (
        2 * 4 * (cur_dim + audio_lstm_cell_size + 1) * audio_lstm_cell_size)
    cur_dim = 2 * audio_lstm_cell_size
  m_audio = float(
      conv1_params + conv2_params + bi_clstm_params + bi_lstm_params)

  # 2. Text Encoder (TtsEncoderV2): 3 Conv1D (5x1, 512) + 1 Bi-LSTM (256/dir).
  text_conv_params = 0
  in_c = text_embedding_dim
  for _ in range(text_num_conv_layers):
    text_conv_params += (
        text_conv_kernel * in_c * text_conv_channels + 3 * text_conv_channels)
    in_c = text_conv_channels
  text_bilstm_params = (
      2 * 4 * (text_conv_channels + text_lstm_cell_size + 1) *
      text_lstm_cell_size)
  m_text = float(text_conv_params + text_bilstm_params)

  # 3. Decoder (PreNet + 2x Uni-LSTM + Linear Mel/EOS + 5-layer PostNet).
  source_dim = 2 * audio_lstm_cell_size
  prenet_params = (
      (num_mel_bins + 1) * decoder_prenet_dim +
      (decoder_prenet_dim + 1) * decoder_prenet_dim)
  dec_lstm_params = 0
  rnn_in = decoder_prenet_dim + source_dim
  for _ in range(decoder_rnn_layers):
    dec_lstm_params += 4 * (rnn_in + decoder_rnn_cell_dim + 1) * (
        decoder_rnn_cell_dim)
    rnn_in = decoder_rnn_cell_dim
  proj_in = decoder_rnn_cell_dim + source_dim
  linear_params = (proj_in + 1) * (num_mel_bins * reduction_factor + 2)
  postnet_params = (
      postnet_kernel * num_mel_bins * postnet_channels +
      (postnet_layers - 2) *
      (postnet_kernel * postnet_channels * postnet_channels) +
      postnet_kernel * postnet_channels * num_mel_bins +
      postnet_layers * 3 * postnet_channels)
  m_dec = float(
      (prenet_params + dec_lstm_params + linear_params) /
      float(max(1, reduction_factor)) + postnet_params)

  # 4. Attention source and query projection matrix sizes (Section 3.3.4).
  atten_source_size = float(source_dim * atten_hidden_dim)
  query_dim = decoder_prenet_dim + decoder_rnn_cell_dim
  atten_query_size = float(query_dim * atten_hidden_dim)

  return {
      'm_audio': m_audio,
      'm_text': m_text,
      'm_dec': m_dec,
      'atten_source_size': atten_source_size,
      'atten_query_size': atten_query_size,
  }


def estimate_flops_and_side_input(
    condition: str = 'single',
    t_x: Optional[float] = None,
    t_y: Optional[float] = None,
    t_z: Optional[float] = None,
    t_int: Optional[float] = None,
    side_text_bytes: Optional[float] = None,
    side_audio_bytes: Optional[float] = None,
    audio_frames: Optional[float] = None,
    text_tokens: Optional[float] = None,
    output_steps: Optional[float] = None,
    interfering_audio_frames: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
  """Analytically computes FLOPS and side-input sizes via Equation 13 of paper.

  In Section 3.3.4 (Equation 13) and Table 3 of the paper:
    FLOPS = M_audio * T_x + M_text * T_y + M_dec * T_z + FLOPS_atten
  where for each attention source, FLOPS_atten multiplies the attention source
  matrix size by T_x / T_y and the query matrix size by T_z.
  """
  if t_x is None and audio_frames is not None:
    t_x = audio_frames
  if t_y is None and text_tokens is not None:
    t_y = text_tokens
  if t_z is None and output_steps is not None:
    t_z = output_steps * 4.0
  if t_int is None and interfering_audio_frames is not None:
    t_int = interfering_audio_frames
  params = compute_analytical_model_params()
  m_audio = params['m_audio']
  m_text = params['m_text']
  m_dec = params['m_dec']
  s_mat = params['atten_source_size']
  q_mat = params['atten_query_size']

  cond = condition.strip().lower()
  # Canonical sequence lengths from the paper's 24 kHz datasets (12.5 ms step)
  default_tx = 590.6
  default_tz = 590.6
  if cond.startswith('multi'):
    default_ty = 97.2
    default_tint = 404.7
    default_audio_bytes = 230000.0
    default_text_bytes = 60.0
  else:
    default_ty = 163.5
    default_tint = 564.3
    default_audio_bytes = 310000.0
    default_text_bytes = 100.0

  tx_val = float(default_tx if t_x is None else t_x)
  ty_val = float(default_ty if t_y is None else t_y)
  tz_val = float(default_tz if t_z is None else t_z)
  tint_val = float(default_tint if t_int is None else t_int)
  audio_bytes = float(
      default_audio_bytes if side_audio_bytes is None else side_audio_bytes)
  text_bytes = float(
      default_text_bytes if side_text_bytes is None else side_text_bytes)

  # Single-source Vanilla-Seq2seq: Audio Encoder + Decoder + 1-source Attention
  atten_vanilla = s_mat * tx_val + q_mat * tz_val
  vanilla_flops = m_audio * tx_val + m_dec * tz_val + atten_vanilla

  # Dual-audio AEC-Seq2seq: 2x Audio Encoder + Decoder + 2-source Attention
  atten_aec = s_mat * (tx_val + tint_val) + 2.0 * q_mat * tz_val
  aec_flops = (
      m_audio * tx_val + m_audio * tint_val + m_dec * tz_val + atten_aec)

  # Proposed TEC: Audio Encoder + Text Encoder + Decoder + 2-source Attention
  atten_tec = s_mat * (tx_val + ty_val) + 2.0 * q_mat * tz_val
  tec_flops = m_audio * tx_val + m_text * ty_val + m_dec * tz_val + atten_tec

  return {
      'AEC-NLMS': {
          'gflops': 0.0,
          'flops': 0.0,
          'side_input_kb': round(audio_bytes / 1000.0, 4),
          'side_input_bytes': audio_bytes,
      },
      'Vanilla-Seq2seq': {
          'gflops': round(vanilla_flops / 1e9, 2),
          'flops': round(vanilla_flops, 2),
          'side_input_kb': 0.0,
          'side_input_bytes': 0.0,
      },
      'AEC-Seq2seq': {
          'gflops': round(aec_flops / 1e9, 2),
          'flops': round(aec_flops, 2),
          'side_input_kb': round(audio_bytes / 1000.0, 4),
          'side_input_bytes': audio_bytes,
      },
      'TEC': {
          'gflops': round(tec_flops / 1e9, 2),
          'flops': round(tec_flops, 2),
          'side_input_kb': round(text_bytes / 1000.0, 4),
          'side_input_bytes': text_bytes,
      },
  }
