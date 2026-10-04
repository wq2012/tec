#!/usr/bin/env python3
"""CLI script to prepare training and evaluation TFRecord datasets for TEC."""

import argparse
import csv
import os
import sys
from typing import List
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tec import data_prep  # noqa: E402


def _load_manifest_csv(
    csv_path: str,
    sample_rate: int = 24000,
) -> List[data_prep.UtteranceRecord]:
  """Loads `UtteranceRecord` items from a manifest CSV file."""
  records = []
  with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
      wav, sr = data_prep.read_wav_file(row['wav_path'])
      if sr != sample_rate:
        raise ValueError(
            f'Expected sample rate {sample_rate} Hz for {row["wav_path"]}, '
            f'got {sr} Hz')
      records.append(
          data_prep.UtteranceRecord(
              utt_id=row['utt_id'],
              waveform=wav,
              transcript=row['transcript'],
              sample_rate=sr))
  return records


def _generate_synthetic_utterances(
    num_clean: int = 8,
    num_interfering: int = 8,
    sample_rate: int = 24000,
    seed: int = 0,
) -> tuple:
  """Generates synthetic clean and interfering utterances for demo/testing."""
  rng = np.random.RandomState(seed)
  clean_utts = []
  for i in range(num_clean):
    length = rng.randint(int(0.2 * sample_rate), int(0.4 * sample_rate))
    t = np.arange(length, dtype=np.float32) / sample_rate
    freq = 200.0 + 30.0 * i
    wav = (0.2 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    clean_utts.append(
        data_prep.UtteranceRecord(
            utt_id=f'clean_{i:04d}',
            waveform=wav,
            transcript=f'set an alarm for {i + 6} am',
            sample_rate=sample_rate))

  interfering_utts = []
  for j in range(num_interfering):
    length = rng.randint(int(0.45 * sample_rate), int(0.7 * sample_rate))
    t = np.arange(length, dtype=np.float32) / sample_rate
    freq = 350.0 + 25.0 * j
    wav = (0.2 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    interfering_utts.append(
        data_prep.UtteranceRecord(
            utt_id=f'tts_{j:04d}',
            waveform=wav,
            transcript=f'currently in mountain view it is {65 + j} degrees',
            sample_rate=sample_rate))
  return clean_utts, interfering_utts


def main():
  parser = argparse.ArgumentParser(
      description='Prepare TFRecord datasets for Textual Echo Cancellation.')
  parser.add_argument(
      '--clean_manifest_csv',
      type=str,
      default='',
      help='Path to CSV with columns utt_id,wav_path,transcript for clean '
      'speech (e.g. LibriTTS).')
  parser.add_argument(
      '--interfering_manifest_csv',
      type=str,
      default='',
      help='Path to CSV with columns utt_id,wav_path,transcript for '
      'interfering TTS speech (e.g. LJSpeech or VCTK).')
  parser.add_argument(
      '--output_tfrecord',
      type=str,
      required=True,
      help='Output TFRecord file path.')
  parser.add_argument(
      '--snr_db',
      type=float,
      default=0.0,
      help='Target SNR in dB between clean and reverberant interfering speech.')
  parser.add_argument(
      '--reverb_rt60',
      type=float,
      default=0.25,
      help='Room impulse response T60 reverberation time in seconds.')
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=24000,
      help='Audio sample rate in Hz (default 24000).')
  parser.add_argument(
      '--generate_synthetic',
      action='store_true',
      help='If set, generate synthetic utterances instead of reading CSVs.')
  parser.add_argument(
      '--num_synthetic',
      type=int,
      default=8,
      help='Number of synthetic utterances to generate when '
      '--generate_synthetic is enabled.')
  parser.add_argument(
      '--seed', type=int, default=0, help='Random seed for pairing and RIRs.')
  args = parser.parse_args()

  if args.generate_synthetic:
    clean_utts, interfering_utts = _generate_synthetic_utterances(
        num_clean=args.num_synthetic,
        num_interfering=args.num_synthetic,
        sample_rate=args.sample_rate,
        seed=args.seed)
  else:
    if not args.clean_manifest_csv or not args.interfering_manifest_csv:
      raise ValueError(
          'Either --generate_synthetic or both --clean_manifest_csv and '
          '--interfering_manifest_csv must be provided.')
    clean_utts = _load_manifest_csv(args.clean_manifest_csv, args.sample_rate)
    interfering_utts = _load_manifest_csv(
        args.interfering_manifest_csv, args.sample_rate)

  num_written = data_prep.prepare_tfrecord_dataset(
      clean_utterances=clean_utts,
      interfering_utterances=interfering_utts,
      output_tfrecord_path=args.output_tfrecord,
      snr_db=args.snr_db,
      reverb_rt60=args.reverb_rt60,
      seed=args.seed)
  print(f'Successfully wrote {num_written} examples to {args.output_tfrecord}')


if __name__ == '__main__':
  main()
