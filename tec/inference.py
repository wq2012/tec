"""Inference utilities for Textual Echo Cancellation models."""

from typing import Dict, Optional, Sequence
from lingvo import compat as tf
from lingvo.core import py_utils
import numpy as np
from tec import configs
from tec import tokenizer as tec_tokenizer


class TecInferenceRunner:
  """Runs inference with a TEC, AEC-Seq2seq, or Vanilla-Seq2seq model."""

  def __init__(
      self,
      model_config_cls=configs.TecSingleInterfering,
      checkpoint_path: Optional[str] = None,
      decode_max_output_frames: Optional[int] = None,
  ):
    self._graph = tf.Graph()
    with self._graph.as_default():
      model_p = model_config_cls().Task()
      if decode_max_output_frames is not None:
        model_p.decoder.decode_max_output_frames = decode_max_output_frames
      self._task = model_p.Instantiate()

      self._mixed_wav_ph = tf.placeholder(
          tf.float32, shape=[None, None], name='mixed_waveform')
      self._mixed_pad_ph = tf.placeholder(
          tf.float32, shape=[None, None], name='mixed_waveform_padding')
      self._int_wav_ph = tf.placeholder(
          tf.float32, shape=[None, None], name='interfering_waveform')
      self._int_pad_ph = tf.placeholder(
          tf.float32, shape=[None, None], name='interfering_waveform_padding')
      self._ids_ph = tf.placeholder(
          tf.int32, shape=[None, None], name='interfering_ids')
      self._id_pad_ph = tf.placeholder(
          tf.float32, shape=[None, None], name='interfering_id_padding')

      wp = self._task.waveform_processor
      mixed_spec = wp.WaveformsToSpectrograms(
          self._mixed_wav_ph, self._mixed_pad_ph)
      int_spec = wp.WaveformsToSpectrograms(
          self._int_wav_ph, self._int_pad_ph)

      batch_size = tf.shape(self._mixed_wav_ph)[0]
      if decode_max_output_frames is None:
        input_batch = py_utils.NestedMap(
            utt_id=tf.fill([batch_size, 1], 'infer_utt'),
            src=py_utils.NestedMap(
                ids=self._ids_ph,
                paddings=self._id_pad_ph,
                source_features=mixed_spec.spectrograms,
                source_feature_paddings=mixed_spec.paddings,
                source_waveforms=self._mixed_wav_ph,
                source_waveform_paddings=self._mixed_pad_ph,
                interfering_features=int_spec.spectrograms,
                interfering_feature_paddings=int_spec.paddings),
            tgt=py_utils.NestedMap(
                features=mixed_spec.spectrograms,
                feature_paddings=mixed_spec.paddings))
        preds = self._task.ComputePredictions(self._task.theta, input_batch)
        pred_wav, pred_pad = wp.SpectrogramsToWaveforms(
            preds.feature_preds,
            mixed_spec.paddings,
            reference_waveforms=self._mixed_wav_ph)
        pred_len = tf.cast(tf.reduce_sum(1.0 - pred_pad, axis=1), tf.int32)
        self._decode_out = py_utils.NestedMap(
            feature_preds=preds.feature_preds,
            feature_paddings=mixed_spec.paddings,
            predicted_waveforms=pred_wav,
            predicted_waveform_lengths=pred_len,
            eos_probs=preds.eos_probs)
      else:
        input_batch = py_utils.NestedMap(
            utt_id=tf.fill([batch_size, 1], 'infer_utt'),
            src=py_utils.NestedMap(
                ids=self._ids_ph,
                paddings=self._id_pad_ph,
                source_features=mixed_spec.spectrograms,
                source_feature_paddings=mixed_spec.paddings,
                source_waveforms=self._mixed_wav_ph,
                source_waveform_paddings=self._mixed_pad_ph,
                interfering_features=int_spec.spectrograms,
                interfering_feature_paddings=int_spec.paddings),
            tgt=py_utils.NestedMap())
        self._decode_out = self._task.Decode(input_batch)
      self._saver = tf.train.Saver()
      self._init_op = tf.global_variables_initializer()

      self._sess = tf.Session(graph=self._graph)
      self._sess.run(self._init_op)
      if checkpoint_path:
        self._saver.restore(self._sess, checkpoint_path)
    self._tokenizer = tec_tokenizer.CharTokenizer(
        vocab_size=model_config_cls.VOCAB_SIZE)

  def predict(
      self,
      mixed_waveforms: Sequence[np.ndarray],
      interfering_texts: Optional[Sequence[str]] = None,
      interfering_waveforms: Optional[Sequence[np.ndarray]] = None,
  ) -> Dict[str, np.ndarray]:
    """Runs echo cancellation on a batch of mixed waveforms."""
    batch_size = len(mixed_waveforms)
    max_samples = max(len(w) for w in mixed_waveforms)
    mixed_arr = np.zeros((batch_size, max_samples), dtype=np.float32)
    mixed_pad = np.ones((batch_size, max_samples), dtype=np.float32)
    for idx, wav in enumerate(mixed_waveforms):
      mixed_arr[idx, :len(wav)] = wav
      mixed_pad[idx, :len(wav)] = 0.0

    if interfering_waveforms is not None:
      max_int = max(len(w) for w in interfering_waveforms)
      int_arr = np.zeros((batch_size, max_int), dtype=np.float32)
      int_pad = np.ones((batch_size, max_int), dtype=np.float32)
      for idx, wav in enumerate(interfering_waveforms):
        int_arr[idx, :len(wav)] = wav
        int_pad[idx, :len(wav)] = 0.0
    else:
      int_arr = mixed_arr
      int_pad = mixed_pad

    if interfering_texts is None:
      interfering_texts = [''] * batch_size
    ids_arr, id_pad = self._tokenizer.batch_encode(interfering_texts)

    feed_dict = {
        self._mixed_wav_ph: mixed_arr,
        self._mixed_pad_ph: mixed_pad,
        self._int_wav_ph: int_arr,
        self._int_pad_ph: int_pad,
        self._ids_ph: ids_arr,
        self._id_pad_ph: id_pad,
    }
    fetches = {
        'feature_preds': self._decode_out.feature_preds,
        'feature_paddings': self._decode_out.feature_paddings,
        'predicted_waveforms': self._decode_out.predicted_waveforms,
        'predicted_waveform_lengths': (
            self._decode_out.predicted_waveform_lengths),
        'eos_probs': self._decode_out.eos_probs,
    }
    return self._sess.run(fetches, feed_dict=feed_dict)

  def close(self):
    self._sess.close()


MODEL_MAP = {
    'TecSingleInterfering': configs.TecSingleInterfering,
    'TecMultiInterfering': configs.TecMultiInterfering,
    'NoSideInputSingleInterfering': configs.NoSideInputSingleInterfering,
    'NoSideInputMultiInterfering': configs.NoSideInputMultiInterfering,
    'AecSingleInterfering': configs.AecSingleInterfering,
    'AecMultiInterfering': configs.AecMultiInterfering,
}


def run_inference(
    mixed_waveform: np.ndarray,
    model_name: str = 'TecSingleInterfering',
    interfering_text: str = '',
    interfering_waveform: Optional[np.ndarray] = None,
    checkpoint_path: Optional[str] = None,
    decode_max_output_frames: Optional[int] = None,
) -> Dict[str, np.ndarray]:
  """Runs echo cancellation inference on a single mixed waveform array."""
  if model_name not in MODEL_MAP:
    raise ValueError(f'Unknown model_name: {model_name}')
  runner = TecInferenceRunner(
      model_config_cls=MODEL_MAP[model_name],
      checkpoint_path=checkpoint_path,
      decode_max_output_frames=decode_max_output_frames,
  )
  try:
    outputs = runner.predict(
        mixed_waveforms=[mixed_waveform],
        interfering_texts=[interfering_text],
        interfering_waveforms=(
            [interfering_waveform]
            if interfering_waveform is not None else None),
    )
    pred_wav = outputs['predicted_waveforms'][0]
    pred_len = int(outputs['predicted_waveform_lengths'][0])
    if pred_len > 0:
      pred_wav = pred_wav[:pred_len]
    return {
        'predicted_mel': outputs['feature_preds'][0],
        'feature_preds': outputs['feature_preds'],
        'predicted_waveform': pred_wav,
        'predicted_waveforms': outputs['predicted_waveforms'],
        'eos_probs': outputs['eos_probs'][0],
    }
  finally:
    runner.close()


def run_inference_on_wav(
    mixed_wav_path: str,
    model_name: str = 'TecSingleInterfering',
    interfering_text: str = '',
    interfering_wav_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    output_wav_path: Optional[str] = None,
    decode_max_output_frames: Optional[int] = None,
) -> Dict[str, np.ndarray]:
  """Runs echo cancellation inference on a WAV file and optionally saves it."""
  from tec import data_prep  # pylint: disable=g-import-not-at-top
  mixed_wav, sr = data_prep.read_wav_file(mixed_wav_path)
  int_wav = None
  if interfering_wav_path:
    int_wav, _ = data_prep.read_wav_file(interfering_wav_path)
  result = run_inference(
      mixed_waveform=mixed_wav,
      model_name=model_name,
      interfering_text=interfering_text,
      interfering_waveform=int_wav,
      checkpoint_path=checkpoint_path,
      decode_max_output_frames=decode_max_output_frames,
  )
  if output_wav_path:
    data_prep.write_wav_file(
        output_wav_path, result['predicted_waveform'], sample_rate=sr)
  result['sample_rate'] = np.asarray(sr, dtype=np.int32)
  return result
