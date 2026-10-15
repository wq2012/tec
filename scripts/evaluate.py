#!/usr/bin/env python3
"""CLI script to evaluate Mel Cepstral Distortion (MCD), WER, and FLOPS."""

import argparse
from concurrent import futures
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Sequence
from absl import app
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tec import aec_model  # noqa: E402
from tec import configs  # noqa: E402
from tec import data_prep  # noqa: E402
from tec import evaluation  # noqa: E402
from tec import waveform_processor  # noqa: E402
from lingvo import compat as tf  # noqa: E402
from lingvo.core import cluster_factory  # noqa: E402
from lingvo.core import py_utils  # noqa: E402

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


def _transcribe_shard_with_audiocpp(
    indexed_waveforms: Sequence[tuple],
    asr_cli_path: str,
    asr_model_path: str,
    threads: int,
    sample_rate: int,
) -> Dict[int, str]:
  """Transcribes a shard of (index, waveform) pairs with `audiocpp_cli`."""
  if not indexed_waveforms:
    return {}
  with tempfile.TemporaryDirectory() as tmp_dir:
    for idx, wav in indexed_waveforms:
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

  return {
      idx: transcripts_by_id.get(f'utt_{idx:05d}', '')
      for idx, _ in indexed_waveforms
  }


def _batch_transcribe_waveforms_with_audiocpp(
    waveforms: Sequence[np.ndarray],
    asr_cli_path: str,
    asr_model_path: str,
    threads: int = 8,
    sample_rate: int = 24000,
) -> List[str]:
  """Transcribes a batch of waveforms using parallel `audiocpp_cli` shards."""
  if not waveforms:
    return []
  indexed = list(enumerate(waveforms))
  num_workers = min(8, max(1, (len(indexed) + 9) // 10))
  if num_workers == 1:
    res_map = _transcribe_shard_with_audiocpp(
        indexed, asr_cli_path, asr_model_path, threads, sample_rate)
    return [res_map.get(i, '') for i in range(len(waveforms))]

  shards = [indexed[i::num_workers] for i in range(num_workers)]
  threads_per_worker = max(2, threads // max(1, num_workers // 2))
  merged: Dict[int, str] = {}
  with futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
    futs = [
        pool.submit(
            _transcribe_shard_with_audiocpp,
            shard,
            asr_cli_path,
            asr_model_path,
            threads_per_worker,
            sample_rate,
        )
        for shard in shards
    ]
    for fut in futs:
      merged.update(fut.result())

  return [merged.get(i, '') for i in range(len(waveforms))]


def evaluate_tfrecord_dataset(
    tfrecord_path: str,
    method: str,
    checkpoint_path: str = '',
    max_examples: int = 0,
    asr_cli_path: str = '',
    asr_model_path: str = '',
    asr_threads: int = 8,
    text_condition: str = 'matched',
    include_per_utterance: bool = False,
) -> Dict[str, object]:
  """Evaluates MCD, WER, Side Input (KB), and GFLOPS on a TFRecord dataset."""
  if text_condition not in ('matched', 'shuffled', 'empty'):
    raise ValueError(
        "text_condition must be 'matched', 'shuffled', or 'empty'.")

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
    clean_n = (
        int(feat['clean_num_samples'].int64_list.value[0])
        if 'clean_num_samples' in feat and
        feat['clean_num_samples'].int64_list.value
        else len(clean_wav))
    int_n = (
        int(feat['interfering_num_samples'].int64_list.value[0])
        if 'interfering_num_samples' in feat and
        feat['interfering_num_samples'].int64_list.value
        else len(int_wav))
    records.append({
        'utt_id': utt_id,
        'clean_wav': clean_wav,
        'int_wav': int_wav,
        'mixed_wav': mixed_wav,
        'clean_txt': clean_txt,
        'int_txt': int_txt,
        'int_ids': int_ids,
        'clean_num_samples': clean_n,
        'interfering_num_samples': int_n,
    })
    if max_examples > 0 and len(records) >= max_examples:
      break

  if not records:
    raise ValueError(f'No records found in {tfrecord_path}')

  mcds: List[float] = []
  side_kb_list: List[float] = []
  side_padded_kb_list: List[float] = []
  tx_list: List[float] = []
  ty_list: List[float] = []
  tz_list: List[float] = []
  eval_waveforms: List[np.ndarray] = []
  refs: List[str] = [r['clean_txt'] for r in records]

  if method in _MODEL_MAP:
    cfg_cls = _MODEL_MAP[method]
    with tf.Graph().as_default(), cluster_factory.SetEval(True):
      task_p = cfg_cls().Task()
      task = task_p.Instantiate()
      wp = task.waveform_processor

      clean_ph = tf.placeholder(tf.float32, [1, None])
      mixed_ph = tf.placeholder(tf.float32, [1, None])
      int_ph = tf.placeholder(tf.float32, [1, None])
      ids_ph = tf.placeholder(tf.int32, [1, None])
      id_pad_ph = tf.placeholder(tf.float32, [1, None])

      pad_1d = tf.zeros_like(clean_ph)

      clean_spec = wp.WaveformsToSpectrograms(clean_ph, pad_1d)
      mixed_spec = wp.WaveformsToSpectrograms(mixed_ph, pad_1d)
      int_spec = wp.WaveformsToSpectrograms(int_ph, pad_1d)

      inp_batch = py_utils.NestedMap(
          utt_id=tf.constant([['eval_utt']]),
          src=py_utils.NestedMap(
              ids=ids_ph,
              paddings=id_pad_ph,
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
        num_recs = len(records)
        shift = max(1, num_recs // 2)
        for idx, rec in enumerate(records):
          if text_condition == 'empty':
            ids = np.array([2], dtype=np.int32)
            id_pad = np.array([1.0], dtype=np.float32)
            txt_bytes = 0
          elif text_condition == 'shuffled':
            other = records[(idx + shift) % num_recs]
            ids = (
                other['int_ids']
                if len(other['int_ids']) > 0
                else np.array([2], dtype=np.int32))
            id_pad = np.zeros_like(ids, dtype=np.float32)
            txt_bytes = len(other['int_txt'].encode('utf-8'))
          else:
            ids = (
                rec['int_ids']
                if len(rec['int_ids']) > 0
                else np.array([2], dtype=np.int32))
            id_pad = np.zeros_like(ids, dtype=np.float32)
            txt_bytes = len(rec['int_txt'].encode('utf-8'))

          ref_mel, pred_mel, pred_wav = sess.run(
              [ref_mel_t, pred_mel_t, pred_wav_t],
              feed_dict={
                  clean_ph: rec['clean_wav'][None, :],
                  mixed_ph: rec['mixed_wav'][None, :],
                  int_ph: rec['int_wav'][None, :],
                  ids_ph: ids[None, :],
                  id_pad_ph: id_pad[None, :],
              })
          mcds.append(evaluation.compute_mcd(ref_mel, pred_mel))
          eval_waveforms.append(pred_wav)
          tx_frames = float(ref_mel.shape[0])
          tx_list.append(tx_frames)
          tz_list.append(np.ceil(tx_frames / 4.0))
          if 'Tec' in method:
            side_kb_list.append(txt_bytes / 1000.0)
            side_padded_kb_list.append(txt_bytes / 1000.0)
            ty_list.append(float(len(ids)))
          elif 'Aec' in method:
            side_kb_list.append(rec['interfering_num_samples'] * 2.0 / 1000.0)
            side_padded_kb_list.append(len(rec['int_wav']) * 2.0 / 1000.0)
            ty_list.append(float(max(1, rec['interfering_num_samples'] // 300)))
          else:
            side_kb_list.append(0.0)
            side_padded_kb_list.append(0.0)
            ty_list.append(0.0)
  else:
    nlms = aec_model.NlmsAec()
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
            side_padded_kb_list.append(0.0)
          elif method == 'MicrophoneSignal':
            eval_wav = rec['mixed_wav']
            pred_mel = sess.run(mel_t, feed_dict={wav_ph: eval_wav[None]})
            side_kb_list.append(0.0)
            side_padded_kb_list.append(0.0)
          elif method == 'NlmsAec':
            eval_wav, _, _ = nlms.process(rec['mixed_wav'], rec['int_wav'])
            pred_mel = sess.run(mel_t, feed_dict={wav_ph: eval_wav[None]})
            side_kb_list.append(rec['interfering_num_samples'] * 2.0 / 1000.0)
            side_padded_kb_list.append(len(rec['int_wav']) * 2.0 / 1000.0)
          else:
            raise ValueError(f'Unknown evaluation method: {method}')

          eval_waveforms.append(eval_wav)
          mcds.append(
              0.0 if method == 'GroundTruth'
              else evaluation.compute_mcd(ref_mel, pred_mel))

  mcd_ci = evaluation.compute_bootstrap_ci(mcds, num_resamples=10000, seed=42)
  out: Dict[str, object] = {
      'method': method,
      'text_condition': text_condition,
      'num_evaluated': len(records),
      'mcd_db': float(np.mean(mcds)),
      'mcd_ci95': mcd_ci,
      'side_input_kb': float(np.mean(side_kb_list)),
      'side_input_padded_kb': float(np.mean(side_padded_kb_list)),
  }
  if method in _MODEL_MAP and tx_list:
    mean_tx = float(np.mean(tx_list))
    mean_ty = float(np.mean(ty_list))
    mean_tz = float(np.mean(tz_list))
    if 'Tec' in method:
      comp = evaluation.estimate_flops_and_side_input(
          audio_frames=mean_tx, text_tokens=mean_ty, output_steps=mean_tz
      )['TEC']
    elif 'Aec' in method:
      comp = evaluation.estimate_flops_and_side_input(
          audio_frames=mean_tx,
          interfering_audio_frames=mean_ty,
          output_steps=mean_tz,
      )['AEC-Seq2seq']
    else:
      comp = evaluation.estimate_flops_and_side_input(
          audio_frames=mean_tx, output_steps=mean_tz
      )['Vanilla-Seq2seq']
    out['gflops'] = comp['gflops']
  elif method == 'NlmsAec':
    mean_samples = float(np.mean([len(r['mixed_wav']) for r in records]))
    out['gflops'] = round(
        (4.0 * float(nlms.filter_length) * mean_samples) / 1e9, 4)
  else:
    out['gflops'] = 0.0

  if asr_cli_path and asr_model_path:
    hyps = _batch_transcribe_waveforms_with_audiocpp(
        eval_waveforms,
        asr_cli_path=asr_cli_path,
        asr_model_path=asr_model_path,
        threads=asr_threads)
    wer_res = evaluation.compute_wer(refs, hyps)
    out['wer'] = wer_res
    utt_err_pct = []
    utt_words = []
    utt_errs = []
    for r_str, h_str in zip(refs, hyps):
      u_res = evaluation.compute_wer([r_str], [h_str])
      w_cnt = max(1, int(u_res['total_words']))
      utt_errs.append(int(u_res['word_errors']))
      utt_err_pct.append(100.0 * float(u_res['word_errors']))
      utt_words.append(float(w_cnt))
    out['wer_ci95'] = evaluation.compute_bootstrap_ci(
        utt_err_pct, weights=utt_words, num_resamples=10000, seed=42)
    if include_per_utterance:
      out['per_utterance_word_errors'] = utt_errs
      out['per_utterance_word_counts'] = [int(w) for w in utt_words]
      out['hypotheses'] = hyps
  if include_per_utterance:
    out['per_utterance_mcds'] = [float(x) for x in mcds]
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
      '--text_condition',
      type=str,
      default='matched',
      choices=['matched', 'shuffled', 'empty'],
      help='Text side-input condition for TEC ablation studies.')
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
        asr_model_path=args.asr_model_path,
        text_condition=args.text_condition)
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
