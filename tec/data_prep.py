"""Dataset preparation pipeline for Textual Echo Cancellation.

Implements the dataset mixing pipeline described in Section 3.1 of the paper
(https://arxiv.org/pdf/2008.06006):
1. Pairs each clean user utterance (e.g. LibriTTS) with a strictly longer
   interfering TTS utterance (e.g. LJSpeech or VCTK).
2. Convolves the interfering TTS waveform with a room impulse response (RIR)
   to simulate room reverberation.
3. Mixes the reverberant interfering speech with the clean speech at a target
   SNR (0 dB in the paper).
4. Pads trailing zeros to the clean speech waveform so its length matches the
   mixed speech waveform.
5. Serializes the prepared examples into TFRecord files.
"""

import dataclasses
import os
from typing import List, Optional, Sequence, Tuple
import wave
from lingvo import compat as tf
import numpy as np
from scipy import signal
from tec import tokenizer as tec_tokenizer


@dataclasses.dataclass
class UtteranceRecord:
  """Represents a single speech utterance with waveform and text transcript."""
  utt_id: str
  waveform: np.ndarray
  transcript: str
  sample_rate: int = 24000


def read_wav_file(wav_path: str) -> Tuple[np.ndarray, int]:
  """Reads a 16-bit or 32-bit PCM WAV file into a float32 array in [-1, 1]."""
  with wave.open(wav_path, 'rb') as wf:
    sample_rate = wf.getframerate()
    num_frames = wf.getnframes()
    num_channels = wf.getnchannels()
    sampwidth = wf.getsampwidth()
    raw_bytes = wf.readframes(num_frames)

  if sampwidth == 2:
    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    samples /= 32768.0
  elif sampwidth == 4:
    samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32)
    samples /= 2147483648.0
  else:
    raise ValueError(f'Unsupported WAV sample width: {sampwidth}')

  if num_channels > 1:
    samples = samples.reshape(-1, num_channels)[:, 0]
  return samples, sample_rate


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
) -> List[Tuple[UtteranceRecord, UtteranceRecord]]:
  """Pairs each clean utterance with a strictly longer interfering utterance."""
  rng = np.random.RandomState(seed)
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
  if len(interfering) < len(clean):
    raise ValueError(
        f'interfering_waveform length ({len(interfering)}) must be >= '
        f'clean_waveform length ({len(clean)})')

  clean_power = np.mean(np.square(clean)) if len(clean) > 0 else 0.0
  interfering_power = (
      np.mean(np.square(interfering)) if len(interfering) > 0 else 0.0)

  if clean_power > 1e-12 and interfering_power > 1e-12:
    target_interfering_power = clean_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_interfering_power / interfering_power)
  else:
    scale = 1.0

  scaled_interfering = interfering * scale
  padded_clean = pad_clean_to_match_mixed(clean, len(interfering))
  mixed = padded_clean + scaled_interfering

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
) -> int:
  """Pairs, reverberates, mixes, and writes utterances to a TFRecord file."""
  pairs = pair_utterances(clean_utterances, interfering_utterances, seed=seed)
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
      combined_id = f'{clean_utt.utt_id}__{int_utt.utt_id}'
      example = create_tf_example(
          utt_id=combined_id,
          clean_waveform=padded_clean_wav,
          interfering_waveform=int_utt.waveform,
          mixed_waveform=mixed_wav,
          clean_transcript=clean_utt.transcript,
          interfering_transcript=int_utt.transcript,
          tokenizer=tokenizer)
      writer.write(example.SerializeToString())
      count += 1
  return count
