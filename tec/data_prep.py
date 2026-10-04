"""Dataset preparation pipeline for Textual Echo Cancellation.

Implements the dataset mixing pipeline described in Sections 3.1 and 3.2 of the
paper (https://arxiv.org/pdf/2008.06006):
1. Builds train/test manifests for LibriTTS, LJSpeech (90%/10% random split),
   and VCTK (90%/10% per-speaker random split across 109 speakers).
2. Pairs each clean user utterance (LibriTTS) with an interfering TTS utterance
   (LJSpeech or VCTK).
3. Convolves the interfering TTS waveform with a room impulse response (RIR)
   to simulate room reverberation.
4. Mixes the reverberant interfering speech with the clean speech at a target
   SNR (0 dB in the paper) and pads the shorter utterance to match lengths.
5. Serializes the prepared examples into TFRecord files.
"""

import csv
import dataclasses
import math
import os
from typing import List, Optional, Sequence, Tuple
import wave
import numpy as np
from scipy import signal
from scipy.io import wavfile
from tec import tokenizer as tec_tokenizer

try:
  from lingvo import compat as tf
except ImportError:
  import tensorflow.compat.v1 as tf  # type: ignore


@dataclasses.dataclass
class UtteranceRecord:
  """Represents a single speech utterance with waveform and text transcript."""
  utt_id: str
  waveform: np.ndarray
  transcript: str
  sample_rate: int = 24000


def resample_waveform(
    waveform: np.ndarray,
    orig_sample_rate: int,
    target_sample_rate: int = 24000,
) -> np.ndarray:
  """Resamples a 1D waveform from `orig_sample_rate` to `target_sample_rate`."""
  wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
  if orig_sample_rate == target_sample_rate or len(wav) == 0:
    return wav
  divisor = math.gcd(int(orig_sample_rate), int(target_sample_rate))
  up = int(target_sample_rate) // divisor
  down = int(orig_sample_rate) // divisor
  resampled = signal.resample_poly(wav, up, down)
  return np.asarray(resampled, dtype=np.float32)


def read_wav_file(
    wav_path: str,
    target_sample_rate: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
  """Reads a PCM or IEEE-float WAV file into a float32 array in [-1, 1]."""
  sample_rate, data = wavfile.read(wav_path)
  if data.ndim > 1:
    data = data[:, 0]

  if data.dtype == np.int16:
    samples = data.astype(np.float32) / 32768.0
  elif data.dtype == np.int32:
    samples = data.astype(np.float32) / 2147483648.0
  elif data.dtype == np.uint8:
    samples = (data.astype(np.float32) - 128.0) / 128.0
  elif np.issubdtype(data.dtype, np.floating):
    samples = data.astype(np.float32)
  else:
    raise ValueError(f'Unsupported WAV dtype: {data.dtype}')

  if target_sample_rate is not None and sample_rate != target_sample_rate:
    samples = resample_waveform(samples, int(sample_rate), target_sample_rate)
    sample_rate = target_sample_rate
  return samples, int(sample_rate)


def write_wav_file(
    wav_path: str,
    waveform: np.ndarray,
    sample_rate: int = 24000,
) -> None:
  """Writes a float32 waveform in [-1, 1] to a 16-bit PCM WAV file."""
  clipped = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
  pcm16 = (clipped * 32767.0).astype(np.int16)
  os.makedirs(os.path.dirname(os.path.abspath(wav_path)), exist_ok=True)
  with wave.open(wav_path, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(int(sample_rate))
    wf.writeframes(pcm16.tobytes())


def pair_utterances(
    clean_utterances: Sequence[UtteranceRecord],
    interfering_utterances: Sequence[UtteranceRecord],
    seed: int = 0,
    require_longer_interfering: bool = True,
) -> List[Tuple[UtteranceRecord, UtteranceRecord]]:
  """Pairs each clean utterance with an interfering utterance."""
  if not interfering_utterances:
    return []
  rng = np.random.RandomState(seed)

  if not require_longer_interfering:
    pairs = []
    for clean_utt in clean_utterances:
      chosen_idx = rng.randint(0, len(interfering_utterances))
      pairs.append((clean_utt, interfering_utterances[chosen_idx]))
    return pairs

  sorted_interfering = sorted(
      interfering_utterances, key=lambda u: len(u.waveform))
  interfering_lengths = [len(u.waveform) for u in sorted_interfering]

  pairs = []
  for clean_utt in clean_utterances:
    clean_len = len(clean_utt.waveform)
    start_idx = int(
        np.searchsorted(interfering_lengths, clean_len, side='right'))
    if start_idx >= len(sorted_interfering):
      continue
    chosen_idx = rng.randint(start_idx, len(sorted_interfering))
    pairs.append((clean_utt, sorted_interfering[chosen_idx]))
  return pairs


def generate_synthetic_rir(
    sample_rate: int = 24000,
    rt60: float = 0.25,
    duration_sec: float = 0.25,
    direct_delay_ms: float = 5.0,
    seed: Optional[int] = None,
) -> np.ndarray:
  """Generates an exponentially decaying room impulse response (RIR)."""
  rng = np.random.RandomState(seed)
  num_samples = max(1, int(round(duration_sec * sample_rate)))
  direct_idx = min(
      num_samples - 1, max(0, int(round(direct_delay_ms * sample_rate / 1000))))

  rir = np.zeros(num_samples, dtype=np.float64)
  rir[direct_idx] = 1.0

  if rt60 > 0 and direct_idx + 1 < num_samples:
    t = np.arange(num_samples - direct_idx - 1, dtype=np.float64) / sample_rate
    decay_rate = np.log(10.0) * 3.0 / rt60
    envelope = np.exp(-decay_rate * t)
    rir[direct_idx + 1:] = rng.normal(0.0, 0.3, size=len(t)) * envelope

  norm = np.linalg.norm(rir)
  if norm > 0:
    rir /= norm
  return rir.astype(np.float32)


def apply_reverberation(waveform: np.ndarray, rir: np.ndarray) -> np.ndarray:
  """Convolves an audio waveform with a room impulse response (RIR)."""
  wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
  rir_arr = np.asarray(rir, dtype=np.float32).reshape(-1)
  if len(wav) == 0 or len(rir_arr) == 0:
    return wav.copy()
  convolved = signal.fftconvolve(wav, rir_arr, mode='full')[:len(wav)]
  return convolved.astype(np.float32)


def pad_clean_to_match_mixed(
    clean_waveform: np.ndarray,
    target_length: int,
) -> np.ndarray:
  """Pads trailing zeros to `clean_waveform` to reach `target_length`."""
  clean = np.asarray(clean_waveform, dtype=np.float32).reshape(-1)
  if target_length < len(clean):
    raise ValueError(
        f'target_length ({target_length}) must be >= len(clean) ({len(clean)})')
  if target_length == len(clean):
    return clean.copy()
  return np.pad(clean, (0, target_length - len(clean)), mode='constant')


def mix_waveforms_at_snr(
    clean_waveform: np.ndarray,
    interfering_waveform: np.ndarray,
    snr_db: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
  """Mixes clean and interfering waveforms at a specified SNR (in dB)."""
  clean = np.asarray(clean_waveform, dtype=np.float64).reshape(-1)
  interfering = np.asarray(interfering_waveform, dtype=np.float64).reshape(-1)

  clean_power = np.mean(np.square(clean)) if len(clean) > 0 else 0.0
  interfering_power = (
      np.mean(np.square(interfering)) if len(interfering) > 0 else 0.0)

  if clean_power > 1e-12 and interfering_power > 1e-12:
    target_interfering_power = clean_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_interfering_power / interfering_power)
  else:
    scale = 1.0

  scaled_interfering = interfering * scale
  target_len = max(len(clean), len(scaled_interfering))
  padded_clean = pad_clean_to_match_mixed(clean, target_len)
  padded_interfering = pad_clean_to_match_mixed(scaled_interfering, target_len)
  mixed = padded_clean + padded_interfering

  max_abs = max(np.max(np.abs(mixed)), np.max(np.abs(padded_clean)), 1e-8)
  if max_abs > 0.99:
    norm_factor = 0.99 / max_abs
    mixed *= norm_factor
    padded_clean *= norm_factor

  return mixed.astype(np.float32), padded_clean.astype(np.float32)


def create_tf_example(
    utt_id: str,
    clean_waveform: np.ndarray,
    interfering_waveform: np.ndarray,
    mixed_waveform: np.ndarray,
    clean_transcript: str,
    interfering_transcript: str,
    tokenizer: Optional[tec_tokenizer.CharTokenizer] = None,
) -> tf.train.Example:
  """Builds a `tf.train.Example` proto for `TecInputGenerator`."""
  if tokenizer is None:
    tokenizer = tec_tokenizer.CharTokenizer()
  interfering_ids = tokenizer.text_to_ids(interfering_transcript)

  feature = {
      'utt_id': tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[utt_id.encode('utf-8')])),
      'clean_waveform': tf.train.Feature(
          float_list=tf.train.FloatList(
              value=np.asarray(clean_waveform, dtype=np.float32).reshape(-1))),
      'interfering_waveform': tf.train.Feature(
          float_list=tf.train.FloatList(
              value=np.asarray(
                  interfering_waveform, dtype=np.float32).reshape(-1))),
      'mixed_waveform': tf.train.Feature(
          float_list=tf.train.FloatList(
              value=np.asarray(mixed_waveform, dtype=np.float32).reshape(-1))),
      'clean_transcript': tf.train.Feature(
          bytes_list=tf.train.BytesList(
              value=[clean_transcript.encode('utf-8')])),
      'interfering_transcript': tf.train.Feature(
          bytes_list=tf.train.BytesList(
              value=[interfering_transcript.encode('utf-8')])),
      'interfering_ids': tf.train.Feature(
          int64_list=tf.train.Int64List(value=interfering_ids)),
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))


def prepare_tfrecord_dataset(
    clean_utterances: Sequence[UtteranceRecord],
    interfering_utterances: Sequence[UtteranceRecord],
    output_tfrecord_path: str,
    snr_db: float = 0.0,
    reverb_rt60: float = 0.25,
    seed: int = 0,
    require_longer_interfering: bool = False,
) -> int:
  """Pairs, reverberates, mixes, and writes utterances to a TFRecord file."""
  pairs = pair_utterances(
      clean_utterances,
      interfering_utterances,
      seed=seed,
      require_longer_interfering=require_longer_interfering)
  tokenizer = tec_tokenizer.CharTokenizer()
  os.makedirs(
      os.path.dirname(os.path.abspath(output_tfrecord_path)), exist_ok=True)

  count = 0
  with tf.io.TFRecordWriter(output_tfrecord_path) as writer:
    for idx, (clean_utt, int_utt) in enumerate(pairs):
      rir = generate_synthetic_rir(
          sample_rate=clean_utt.sample_rate,
          rt60=reverb_rt60,
          seed=seed + idx + 1)
      reverb_int_wav = apply_reverberation(int_utt.waveform, rir)
      mixed_wav, padded_clean_wav = mix_waveforms_at_snr(
          clean_utt.waveform, reverb_int_wav, snr_db=snr_db)
      padded_int_wav = pad_clean_to_match_mixed(
          int_utt.waveform, len(mixed_wav))
      combined_id = f'{clean_utt.utt_id}__{int_utt.utt_id}'
      example = create_tf_example(
          utt_id=combined_id,
          clean_waveform=padded_clean_wav,
          interfering_waveform=padded_int_wav,
          mixed_waveform=mixed_wav,
          clean_transcript=clean_utt.transcript,
          interfering_transcript=int_utt.transcript,
          tokenizer=tokenizer)
      writer.write(example.SerializeToString())
      count += 1
  return count


def _write_manifest_rows(
    csv_path: str,
    rows: Sequence[Tuple[str, str, str]],
) -> int:
  """Writes `(utt_id, wav_path, transcript)` rows to a CSV file."""
  os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
  with open(csv_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['utt_id', 'wav_path', 'transcript'])
    for row in rows:
      writer.writerow(row)
  return len(rows)


def build_ljspeech_manifests(
    ljspeech_root: str,
    train_csv_path: str,
    test_csv_path: str,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[int, int]:
  """Builds 90%/10% train/test CSV manifests for LJSpeech (Paper Sec 3.1)."""
  meta_path = os.path.join(ljspeech_root, 'metadata.csv')
  wav_dir = os.path.join(ljspeech_root, 'wavs')
  if not os.path.isfile(meta_path):
    nested = os.path.join(ljspeech_root, 'LJSpeech-1.1')
    if os.path.isfile(os.path.join(nested, 'metadata.csv')):
      meta_path = os.path.join(nested, 'metadata.csv')
      wav_dir = os.path.join(nested, 'wavs')

  rows = []
  with open(meta_path, 'r', encoding='utf-8') as f:
    for line in f:
      parts = line.strip().split('|')
      if len(parts) < 2:
        continue
      utt_id = parts[0].strip()
      transcript = parts[-1].strip()
      wav_path = os.path.join(wav_dir, f'{utt_id}.wav')
      if os.path.isfile(wav_path) and transcript:
        rows.append((utt_id, wav_path, transcript))

  rng = np.random.RandomState(seed)
  indices = rng.permutation(len(rows))
  split_idx = int(round(len(rows) * train_ratio))
  train_rows = [rows[i] for i in sorted(indices[:split_idx])]
  test_rows = [rows[i] for i in sorted(indices[split_idx:])]
  return (
      _write_manifest_rows(train_csv_path, train_rows),
      _write_manifest_rows(test_csv_path, test_rows),
  )


def build_vctk_manifests(
    vctk_root: str,
    train_csv_path: str,
    test_csv_path: str,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[int, int]:
  """Builds 90%/10% per-speaker train/test CSV manifests for VCTK."""
  rng = np.random.RandomState(seed)
  train_rows = []
  test_rows = []

  speaker_dirs = sorted(
      d for d in os.listdir(vctk_root)
      if os.path.isdir(os.path.join(vctk_root, d)))
  for spk in speaker_dirs:
    spk_dir = os.path.join(vctk_root, spk)
    spk_rows = []
    for fname in sorted(os.listdir(spk_dir)):
      if not fname.endswith('.wav'):
        continue
      utt_id = os.path.splitext(fname)[0]
      txt_path = os.path.join(spk_dir, f'{utt_id}.txt')
      wav_path = os.path.join(spk_dir, fname)
      if not os.path.isfile(txt_path):
        continue
      with open(txt_path, 'r', encoding='utf-8', errors='ignore') as tf_in:
        transcript = tf_in.read().strip()
      if transcript:
        spk_rows.append((utt_id, wav_path, transcript))

    if not spk_rows:
      continue
    indices = rng.permutation(len(spk_rows))
    split_idx = int(round(len(spk_rows) * train_ratio))
    for i in sorted(indices[:split_idx]):
      train_rows.append(spk_rows[i])
    for i in sorted(indices[split_idx:]):
      test_rows.append(spk_rows[i])

  return (
      _write_manifest_rows(train_csv_path, train_rows),
      _write_manifest_rows(test_csv_path, test_rows),
  )


def build_libritts_manifest(
    split_dirs: Sequence[str],
    output_csv_path: str,
) -> int:
  """Builds a CSV manifest from one or more LibriTTS split directories."""
  rows = []
  for split_dir in split_dirs:
    for root, _, files in os.walk(split_dir):
      file_set = set(files)
      for fname in sorted(files):
        if not fname.endswith('.wav'):
          continue
        utt_id = os.path.splitext(fname)[0]
        norm_txt = f'{utt_id}.normalized.txt'
        orig_txt = f'{utt_id}.original.txt'
        txt_file = norm_txt if norm_txt in file_set else orig_txt
        if txt_file not in file_set:
          continue
        txt_path = os.path.join(root, txt_file)
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as tf_in:
          transcript = tf_in.read().strip()
        if transcript:
          rows.append((utt_id, os.path.join(root, fname), transcript))
  return _write_manifest_rows(output_csv_path, rows)
