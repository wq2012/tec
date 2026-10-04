#!/usr/bin/env python3
"""CLI script to evaluate Mel Cepstral Distortion (MCD), WER, and FLOPS."""

import argparse
import json
import os
import sys
from lingvo import compat as tf
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tec import data_prep  # noqa: E402
from tec import evaluation  # noqa: E402
from tec import waveform_processor  # noqa: E402


def compute_mcd_for_wav_files(ref_wav_path: str, pred_wav_path: str) -> float:
  """Computes MCD (in dB) between two 24 kHz WAV files."""
  ref_wav, _ = data_prep.read_wav_file(ref_wav_path)
  pred_wav, _ = data_prep.read_wav_file(pred_wav_path)

  with tf.Graph().as_default():
    wp = waveform_processor.WaveformProcessor.Params().Instantiate()
    ref_t = tf.constant(ref_wav[None, :], dtype=tf.float32)
    pred_t = tf.constant(pred_wav[None, :], dtype=tf.float32)
    ref_mel_t = wp.WaveformsToSpectrograms(ref_t).spectrograms[0]
    pred_mel_t = wp.WaveformsToSpectrograms(pred_t).spectrograms[0]
    with tf.Session() as sess:
      ref_mel, pred_mel = sess.run([ref_mel_t, pred_mel_t])

  return evaluation.compute_mcd(ref_mel, pred_mel)


def main():
  parser = argparse.ArgumentParser(
      description='Evaluate MCD, WER, and model complexity for TEC.')
  parser.add_argument(
      '--ref_wav',
      type=str,
      default='',
      help='Path to reference clean WAV file for MCD evaluation.')
  parser.add_argument(
      '--pred_wav',
      type=str,
      default='',
      help='Path to predicted WAV file for MCD evaluation.')
  parser.add_argument(
      '--ref_mel_npy',
      type=str,
      default='',
      help='Path to reference log-Mel .npy file for MCD evaluation.')
  parser.add_argument(
      '--pred_mel_npy',
      type=str,
      default='',
      help='Path to predicted log-Mel .npy file for MCD evaluation.')
  parser.add_argument(
      '--ref_transcript',
      type=str,
      default='',
      help='Reference transcript string for WER evaluation.')
  parser.add_argument(
      '--hyp_transcript',
      type=str,
      default='',
      help='Hypothesis transcript string for WER evaluation.')
  parser.add_argument(
      '--print_complexity',
      action='store_true',
      help='Print FLOPS and side-input bandwidth estimates (Table 4).')
  args = parser.parse_args()

  results = {}
  if args.ref_wav and args.pred_wav:
    results['mcd_db'] = compute_mcd_for_wav_files(args.ref_wav, args.pred_wav)
  elif args.ref_mel_npy and args.pred_mel_npy:
    ref_mel = np.load(args.ref_mel_npy)
    pred_mel = np.load(args.pred_mel_npy)
    results['mcd_db'] = evaluation.compute_mcd(ref_mel, pred_mel)

  if args.ref_transcript or args.hyp_transcript:
    results['wer'] = evaluation.compute_wer(
        [args.ref_transcript], [args.hyp_transcript])

  if args.print_complexity:
    results['complexity'] = evaluation.estimate_flops_and_side_input()

  print(json.dumps(results, indent=2))


if __name__ == '__main__':
  main()
