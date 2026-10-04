"""Evaluation metrics for Textual Echo Cancellation (MCD, DTW, WER, FLOPS)."""

import re
from typing import Dict, Sequence, Tuple
import numpy as np
from scipy import fftpack

_PUNCTUATION_CHARS = r'!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~'
_PUNCTUATION_RE = re.compile(f'[{_PUNCTUATION_CHARS}]')


def compute_mfcc_from_log_mel(
    log_mel_spectrogram: np.ndarray,
    num_mfccs: int = 13,
) -> np.ndarray:
  """Computes MFCCs from a log-Mel spectrogram via type-II DCT."""
  mel = np.asarray(log_mel_spectrogram, dtype=np.float64)
  if mel.ndim != 2:
    raise ValueError(
        f'Expected 2D log_mel_spectrogram [frames, bins], got {mel.shape}')
  num_bins = mel.shape[1]
  dct_all = fftpack.dct(mel, type=2, axis=-1, norm=None)
  scale = 0.5 / np.sqrt(float(num_bins) / 2.0)
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

  cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
  steps = np.zeros((n + 1, m + 1), dtype=np.int32)
  cost[0, 0] = 0.0

  for i in range(1, n + 1):
    for j in range(1, m + 1):
      prev_costs = (cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1])
      prev_steps = (steps[i - 1, j - 1], steps[i - 1, j], steps[i, j - 1])
      best_idx = int(np.argmin(prev_costs))
      cost[i, j] = dist_matrix[i - 1, j - 1] + prev_costs[best_idx]
      steps[i, j] = prev_steps[best_idx] + 1

  return float(cost[n, m]), int(steps[n, m])


def compute_mcd(
    log_mel_ref: np.ndarray,
    log_mel_pred: np.ndarray,
    num_mfccs: int = 13,
    use_dtw: bool = True,
) -> float:
  """Computes Mel Cepstral Distortion (MCD) in dB between two spectrograms.

  Matches the MCD metric implementation from Section 3.3.2 (Equation 11) of the
  paper (excluding the 0th cepstral coefficient c_0, using 13 MFCCs with DTW
  time alignment, and normalizing the DTW Euclidean distance by
  max(T_ref, T_pred)).
  """
  mfcc_ref = compute_mfcc_from_log_mel(log_mel_ref, num_mfccs=num_mfccs)[:, 1:]
  mfcc_pred = compute_mfcc_from_log_mel(
      log_mel_pred, num_mfccs=num_mfccs)[:, 1:]

  if use_dtw:
    total_dist, _ = compute_dtw_distance(mfcc_ref, mfcc_pred)
    return float(total_dist / max(len(mfcc_ref), len(mfcc_pred)))

  min_len = min(len(mfcc_ref), len(mfcc_pred))
  frame_dists = np.linalg.norm(
      mfcc_ref[:min_len] - mfcc_pred[:min_len], axis=-1)
  return float(np.mean(frame_dists))


def normalize_transcript(text: str) -> str:
  """Strips punctuation and normalizes whitespace and case for WER scoring."""
  cleaned = _PUNCTUATION_RE.sub('', text)
  return re.sub(r'\s+', ' ', cleaned).strip().lower()


def compute_wer(
    references: Sequence[str],
    hypotheses: Sequence[str],
    normalize: bool = True,
) -> Dict[str, float]:
  """Computes Word Error Rate (WER) across reference and hypothesis pairs."""
  if len(references) != len(hypotheses):
    raise ValueError(
        f'Length mismatch: {len(references)} references vs '
        f'{len(hypotheses)} hypotheses')

  total_edits = 0
  total_words = 0

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
    total_edits += int(dp[n, m])

  wer_pct = (100.0 * total_edits / total_words) if total_words > 0 else 0.0
  return {
      'wer': float(wer_pct),
      'word_errors': int(total_edits),
      'total_words': int(total_words),
  }


def estimate_flops_and_side_input(
    condition: str = 'single',
) -> Dict[str, Dict[str, float]]:
  """Returns FLOPS and side-input sizes from Section 3.3.4 & Table 3 of paper.

  In Section 3.3.4 (Equation 12) and Table 3 of the paper:
    FLOPS = M_audio * T_x + M_text * T_y + M_dec * T_z + FLOPS_atten

  Args:
    condition: Either `'single'` (Single interfering voice: LibriTTS +
      LJ Speech, average echo ~7s) or `'multi'` (Multiple interfering voices:
      LibriTTS + VCTK, average echo ~2s).

  Returns:
    Dictionary mapping each method (`'AEC-NLMS'`, `'Vanilla-Seq2seq'`,
    `'AEC-Seq2seq'`, `'TEC'`) to `'gflops'`, `'flops'`, `'side_input_kb'`, and
    `'side_input_bytes'`.
  """
  cond = condition.strip().lower()
  if cond.startswith('multi'):
    audio_side_kb = 230.0
    text_side_kb = 0.06
    vanilla_gflops = 6.32
    aec_gflops = 8.62
    tec_gflops = 6.90
  else:
    audio_side_kb = 310.0
    text_side_kb = 0.10
    vanilla_gflops = 6.32
    aec_gflops = 9.51
    tec_gflops = 7.27

  return {
      'AEC-NLMS': {
          'gflops': 0.0,
          'flops': 0.0,
          'side_input_kb': audio_side_kb,
          'side_input_bytes': audio_side_kb * 1000.0,
      },
      'Vanilla-Seq2seq': {
          'gflops': vanilla_gflops,
          'flops': vanilla_gflops * 1e9,
          'side_input_kb': 0.0,
          'side_input_bytes': 0.0,
      },
      'AEC-Seq2seq': {
          'gflops': aec_gflops,
          'flops': aec_gflops * 1e9,
          'side_input_kb': audio_side_kb,
          'side_input_bytes': audio_side_kb * 1000.0,
      },
      'TEC': {
          'gflops': tec_gflops,
          'flops': tec_gflops * 1e9,
          'side_input_kb': text_side_kb,
          'side_input_bytes': text_side_kb * 1000.0,
      },
  }
