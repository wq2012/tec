#!/usr/bin/env python3
"""CLI script to evaluate Mel Cepstral Distortion (MCD), WER, and FLOPS."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Sequence
from absl import app
from lingvo import compat as tf
from lingvo.core import py_utils
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tec import aec_model  # noqa: E402
from tec import configs  # noqa: E402
from tec import data_prep  # noqa: E402
from tec import evaluation  # noqa: E402
from tec import waveform_processor  # noqa: E402

_MODEL_MAP = {
    'TecSingleInterfering': configs.TecSingleInterfering,
    'TecMultiInterfering': configs.TecMultiInterfering,
    'NoSideInputSingleInterfering': configs.NoSideInputSingleInterfering,
    'NoSideInputMultiInterfering': configs.NoSideInputMultiInterfering,
    'AecSingleInterfering': configs.AecSingleInterfering,
    'AecMultiInterfering': configs.AecMultiInterfering,
}


def compute_mcd_for_wav_files(ref_wav_path: str, pred_wav_path: str) -> float:
  """Computes MCD (in dB) between two 24 kHz WAV files."""
  ref_wav, _ = data_prep.read_wav_file(ref_wav_path, target_sample_rate=24000)
  pred_wav, _ = data_prep.read_wav_file(pred_wav_path, target_sample_rate=24000)

  with tf.Graph().as_default():
    wp = waveform_processor.WaveformProcessor.Params().Instantiate()
    ref_t = tf.constant(ref_wav[None, :], dtype=tf.float32)
    pred_t = tf.constant(pred_wav[None, :], dtype=tf.float32)
    ref_mel_t = wp.WaveformsToSpectrograms(ref_t).spectrograms[0]
    pred_mel_t = wp.WaveformsToSpectrograms(pred_t).spectrograms[0]
    with tf.Session() as sess:
      ref_mel, pred_mel = sess.run([ref_mel_t, pred_mel_t])

  return evaluation.compute_mcd(ref_mel, pred_mel)


def _batch_transcribe_waveforms_with_audiocpp(
    waveforms: Sequence[np.ndarray],
    asr_cli_path: str,
    asr_model_path: str,
    threads: int = 8,
    sample_rate: int = 24000,
) -> List[str]:
  """Transcribes a list of waveforms in a single batch call to `audiocpp_cli`."""
  if not waveforms:
    return []
  with tempfile.TemporaryDirectory() as tmp_dir:
    for idx, wav in enumerate(waveforms):
      wav_path = os.path.join(tmp_dir, f'utt_{idx:05d}.wav')
      data_prep.write_wav_file(wav_path, wav, sample_rate=sample_rate)

    cmd = [
        asr_cli_path,
        '--task', 'asr',
        '--family', 'qwen3_asr',
        '--model', asr_model_path,
        '--backend', 'cpu',
        '--threads', str(threads),
        '--batch-audio-dir', tmp_dir,
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True)

  transcripts_by_id: Dict[str, str] = {}
  current_id: Optional[str] = None
  for line in proc.stdout.splitlines():
    if line.startswith('request_id='):
      current_id = line[len('request_id='):].strip()
    elif line.startswith('text_output='):
      txt = line[len('text_output='):].strip()
      if current_id is not None:
        transcripts_by_id[current_id] = txt

  return [
      transcripts_by_id.get(f'utt_{idx:05d}', '')
      for idx in range(len(waveforms))
  ]


def evaluate_tfrecord_dataset(
    tfrecord_path: str,
    method: str,
    checkpoint_path: str = '',
    max_examples: int = 0,
    asr_cli_path: str = '',
    asr_model_path: str = '',
    asr_threads: int = 8,
) -> Dict[str, object]:
  """Evaluates MCD, WER, Side Input (KB), and GFLOPS on a TFRecord dataset."""
  records = []
  for raw_bytes in tf.io.tf_record_iterator(tfrecord_path):
    ex = tf.train.Example()
    ex.ParseFromString(raw_bytes)
    feat = ex.features.feature
    utt_id = feat['utt_id'].bytes_list.value[0].decode('utf-8')
    clean_wav = np.array(
        feat['clean_waveform'].float_list.value, dtype=np.float32)
    int_wav = np.array(
        feat['interfering_waveform'].float_list.value, dtype=np.float32)
    mixed_wav = np.array(
        feat['mixed_waveform'].float_list.value, dtype=np.float32)
    clean_txt = feat['clean_transcript'].bytes_list.value[0].decode('utf-8')
    int_txt = feat['interfering_transcript'].bytes_list.value[0].decode('utf-8')
    int_ids = np.array(
        feat['interfering_ids'].int64_list.value, dtype=np.int32)
    records.append({
        'utt_id': utt_id,
        'clean_wav': clean_wav,
        'int_wav': int_wav,
        'mixed_wav': mixed_wav,
        'clean_txt': clean_txt,
        'int_txt': int_txt,
        'int_ids': int_ids,
    })
    if max_examples > 0 and len(records) >= max_examples:
      break

  if not records:
    raise ValueError(f'No records found in {tfrecord_path}')

  mcds: List[float] = []
  side_kb_list: List[float] = []
  eval_waveforms: List[np.ndarray] = []
  refs: List[str] = [r['clean_txt'] for r in records]

  if method in _MODEL_MAP:
    cfg_cls = _MODEL_MAP[method]
    with tf.Graph().as_default():
      task_p = cfg_cls().Task()
      task = task_p.Instantiate()
      wp = task.waveform_processor

      clean_ph = tf.placeholder(tf.float32, [1, None])
      mixed_ph = tf.placeholder(tf.float32, [1, None])
      int_ph = tf.placeholder(tf.float32, [1, None])
      ids_ph = tf.placeholder(tf.int32, [1, None])

      pad_1d = tf.zeros_like(clean_ph)
      id_pad = tf.zeros_like(ids_ph, dtype=tf.float32)

      clean_spec = wp.WaveformsToSpectrograms(clean_ph, pad_1d)
      mixed_spec = wp.WaveformsToSpectrograms(mixed_ph, pad_1d)
      int_spec = wp.WaveformsToSpectrograms(int_ph, pad_1d)

      inp_batch = py_utils.NestedMap(
          utt_id=tf.constant([['eval_utt']]),
          src=py_utils.NestedMap(
              ids=ids_ph,
              paddings=id_pad,
              source_features=mixed_spec.spectrograms,
              source_feature_paddings=mixed_spec.paddings,
              source_waveforms=mixed_ph,
              source_waveform_paddings=pad_1d,
              interfering_features=int_spec.spectrograms,
              interfering_feature_paddings=int_spec.paddings),
          tgt=py_utils.NestedMap(
              features=mixed_spec.spectrograms,
              feature_paddings=mixed_spec.paddings))
      preds = task.ComputePredictions(task.theta, inp_batch)
      pred_wav_t = wp.SpectrogramsToWaveforms(
          preds.feature_preds,
          mixed_spec.paddings,
          reference_waveforms=mixed_ph)[0][0]
      pred_mel_t = wp.WaveformsToSpectrograms(
          tf.expand_dims(pred_wav_t, axis=0)).spectrograms[0]
      ref_mel_t = clean_spec.spectrograms[0]
      saver = tf.train.Saver()

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if checkpoint_path:
          saver.restore(sess, checkpoint_path)
        for rec in records:
          ids = rec['int_ids'] if len(rec['int_ids']) > 0 else np.zeros(
              (1,), dtype=np.int32)
          ref_mel, pred_mel, pred_wav = sess.run(
              [ref_mel_t, pred_mel_t, pred_wav_t],
              feed_dict={
                  clean_ph: rec['clean_wav'][None, :],
                  mixed_ph: rec['mixed_wav'][None, :],
                  int_ph: rec['int_wav'][None, :],
                  ids_ph: ids[None, :],
              })
          mcds.append(evaluation.compute_mcd(ref_mel, pred_mel))
          eval_waveforms.append(pred_wav)
          if 'Tec' in method:
            side_kb_list.append(len(rec['int_txt'].encode('utf-8')) / 1000.0)
          elif 'Aec' in method:
            side_kb_list.append(len(rec['int_wav']) * 2.0 / 1000.0)
          else:
            side_kb_list.append(0.0)
  else:
    nlms = aec_model.NlmsAec(filter_length=256, step_size=0.1)
    with tf.Graph().as_default():
      wp = waveform_processor.WaveformProcessor.Params().Instantiate()
      wav_ph = tf.placeholder(tf.float32, [1, None])
      mel_t = wp.WaveformsToSpectrograms(wav_ph).spectrograms[0]
      with tf.Session() as sess:
        for rec in records:
          ref_mel = sess.run(mel_t, feed_dict={wav_ph: rec['clean_wav'][None]})
          if method == 'GroundTruth':
            eval_wav = rec['clean_wav']
            pred_mel = ref_mel
            side_kb_list.append(0.0)
          elif method == 'MicrophoneSignal':
            eval_wav = rec['mixed_wav']
            pred_mel = sess.run(mel_t, feed_dict={wav_ph: eval_wav[None]})
            side_kb_list.append(0.0)
          elif method == 'NlmsAec':
            eval_wav, _, _ = nlms.process(rec['mixed_wav'], rec['int_wav'])
            pred_mel = sess.run(mel_t, feed_dict={wav_ph: eval_wav[None]})
            side_kb_list.append(len(rec['int_wav']) * 2.0 / 1000.0)
          else:
            raise ValueError(f'Unknown evaluation method: {method}')

          eval_waveforms.append(eval_wav)
          mcds.append(
              0.0 if method == 'GroundTruth'
              else evaluation.compute_mcd(ref_mel, pred_mel))

  out: Dict[str, object] = {
      'method': method,
      'num_evaluated': len(records),
      'mcd_db': float(np.mean(mcds)),
      'side_input_kb': float(np.mean(side_kb_list)),
  }
  if asr_cli_path and asr_model_path:
    hyps = _batch_transcribe_waveforms_with_audiocpp(
        eval_waveforms,
        asr_cli_path=asr_cli_path,
        asr_model_path=asr_model_path,
        threads=asr_threads)
    out['wer'] = evaluation.compute_wer(refs, hyps)
  return out


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
      '--eval_tfrecord',
      type=str,
      default='',
      help='Path to a TFRecord dataset for batch evaluation.')
  parser.add_argument(
      '--method',
      type=str,
      default='TecSingleInterfering',
      help='Method to evaluate when --eval_tfrecord is set.')
  parser.add_argument(
      '--checkpoint_path',
      type=str,
      default='',
      help='Checkpoint path when evaluating a trained neural model.')
  parser.add_argument(
      '--max_examples',
      type=int,
      default=0,
      help='Maximum examples to evaluate from --eval_tfrecord (0 = all).')
  parser.add_argument(
      '--asr_cli_path',
      type=str,
      default='',
      help='Optional path to audiocpp_cli binary for ASR WER evaluation.')
  parser.add_argument(
      '--asr_model_path',
      type=str,
      default='',
      help='Optional path to Qwen3-ASR GGUF model for ASR WER evaluation.')
  parser.add_argument(
      '--print_complexity',
      action='store_true',
      help='Print FLOPS and side-input bandwidth estimates.')
  args = parser.parse_args()

  results: Dict[str, Optional[object]] = {}
  if args.eval_tfrecord:
    results = evaluate_tfrecord_dataset(
        tfrecord_path=args.eval_tfrecord,
        method=args.method,
        checkpoint_path=args.checkpoint_path,
        max_examples=args.max_examples,
        asr_cli_path=args.asr_cli_path,
        asr_model_path=args.asr_model_path)
  elif args.ref_wav and args.pred_wav:
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
  app.run(lambda _: main(), argv=sys.argv[:1])
