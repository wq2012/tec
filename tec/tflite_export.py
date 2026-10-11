"""TFLite export and on-device inference for Textual Echo Cancellation."""

import os
from typing import Dict, Optional
from lingvo import compat as tf
from lingvo.core import cluster_factory
from lingvo.core import py_utils
import numpy as np
from tec import aec_model
from tec import configs


def export_to_tflite(
    output_tflite_path: str,
    model_config_cls=configs.TecSingleInterfering,
    checkpoint_path: Optional[str] = None,
    num_frames: int = 200,
    text_length: int = 100,
    decode_steps: int = 200,
    quantize_dynamic_range: bool = True,
) -> bytes:
  """Exports a TEC / AEC / Vanilla-Seq2seq model to a TFLite FlatBuffer."""
  graph = tf.Graph()
  with graph.as_default(), cluster_factory.SetEval(True):
    cfg = model_config_cls()
    task_p = cfg.Task()
    if decode_steps % task_p.decoder.reduction_factor != 0:
      task_p.decoder.reduction_factor = 1
    task_p.decoder.decode_max_output_frames = decode_steps
    task_p.waveform_processor = None
    task = task_p.Instantiate()

    num_mel_bins = cfg.NUM_MEL_BINS
    source_features = tf.placeholder(
        tf.float32,
        shape=[1, num_frames, num_mel_bins],
        name='source_features')
    source_feature_paddings = tf.placeholder(
        tf.float32, shape=[1, num_frames], name='source_feature_paddings')
    interfering_features = tf.placeholder(
        tf.float32,
        shape=[1, num_frames, num_mel_bins],
        name='interfering_features')
    interfering_feature_paddings = tf.placeholder(
        tf.float32,
        shape=[1, num_frames],
        name='interfering_feature_paddings')
    interfering_ids = tf.placeholder(
        tf.int32, shape=[1, text_length], name='interfering_ids')
    interfering_id_paddings = tf.placeholder(
        tf.float32, shape=[1, text_length], name='interfering_id_paddings')

    input_batch = py_utils.NestedMap(
        utt_id=tf.constant([['tflite_utt']]),
        src=py_utils.NestedMap(
            ids=interfering_ids,
            paddings=interfering_id_paddings,
            source_features=source_features,
            source_feature_paddings=source_feature_paddings,
            interfering_features=interfering_features,
            interfering_feature_paddings=interfering_feature_paddings),
        tgt=py_utils.NestedMap(
            features=tf.zeros(
                [1, decode_steps, num_mel_bins], dtype=tf.float32),
            feature_paddings=tf.zeros([1, decode_steps], dtype=tf.float32)))

    predictions = task.ComputePredictions(task.theta, input_batch)
    feature_preds = tf.identity(
        predictions.feature_preds, name='feature_preds')
    eos_probs = tf.identity(predictions.eos_probs, name='eos_probs')

    if isinstance(task, aec_model.AecModel):
      input_tensors = [
          source_features,
          source_feature_paddings,
          interfering_features,
          interfering_feature_paddings,
      ]
    elif task_p.encoder is None:
      input_tensors = [source_features, source_feature_paddings]
    else:
      input_tensors = [
          source_features,
          source_feature_paddings,
          interfering_ids,
          interfering_id_paddings,
      ]
    output_tensors = [feature_preds, eos_probs]

    with tf.Session(graph=graph) as sess:
      sess.run(tf.global_variables_initializer())
      if checkpoint_path:
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)

      converter = tf.compat.v1.lite.TFLiteConverter.from_session(
          sess, input_tensors, output_tensors)
      converter.target_spec.supported_ops = [
          tf.lite.OpsSet.TFLITE_BUILTINS,
          tf.lite.OpsSet.SELECT_TF_OPS,
      ]
      converter.experimental_new_converter = True
      if quantize_dynamic_range:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
      tflite_model = converter.convert()

  if output_tflite_path:
    os.makedirs(
        os.path.dirname(os.path.abspath(output_tflite_path)), exist_ok=True)
    with open(output_tflite_path, 'wb') as f:
      f.write(tflite_model)
  return tflite_model


def _fit_3d_features(
    features: np.ndarray,
    paddings: Optional[np.ndarray],
    target_frames: int,
    target_bins: int,
    pad_value: float = 0.0,
):
  """Shapes 2D/3D Mel features and 1D/2D paddings to `[1, target_frames, D]`."""
  arr = np.asarray(features, dtype=np.float32)
  if arr.ndim == 2:
    arr = arr[None, :, :]
  orig_frames = int(arr.shape[1])
  if paddings is None:
    pad = np.zeros((arr.shape[0], orig_frames), dtype=np.float32)
  else:
    pad = np.asarray(paddings, dtype=np.float32)
    if pad.ndim == 1:
      pad = pad[None, :]
  if arr.shape[2] != target_bins:
    if arr.shape[2] > target_bins:
      arr = arr[:, :, :target_bins]
    else:
      arr = np.pad(
          arr,
          ((0, 0), (0, 0), (0, target_bins - arr.shape[2])),
          constant_values=pad_value)
  if orig_frames > target_frames:
    arr = arr[:, :target_frames, :]
    pad = pad[:, :target_frames]
  elif orig_frames < target_frames:
    extra = target_frames - orig_frames
    arr = np.pad(
        arr, ((0, 0), (0, extra), (0, 0)), constant_values=pad_value)
    pad = np.pad(pad, ((0, 0), (0, extra)), constant_values=1.0)
  return arr, pad, orig_frames


def run_tflite_inference(
    tflite_model_or_path=None,
    source_features: Optional[np.ndarray] = None,
    source_feature_paddings: Optional[np.ndarray] = None,
    interfering_ids: Optional[np.ndarray] = None,
    interfering_id_paddings: Optional[np.ndarray] = None,
    interfering_features: Optional[np.ndarray] = None,
    interfering_feature_paddings: Optional[np.ndarray] = None,
    tflite_path: Optional[str] = None,
    mixed_mel: Optional[np.ndarray] = None,
    interfering_text: Optional[str] = None,
    interfering_mel: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
  """Executes a converted TEC `.tflite` model using `tf.lite.Interpreter`."""
  from tec import tokenizer as tec_tokenizer  # noqa: E402

  model_ref = (
      tflite_model_or_path if tflite_model_or_path is not None else tflite_path)
  if model_ref is None:
    raise ValueError('Either tflite_model_or_path or tflite_path is required.')
  if source_features is None and mixed_mel is not None:
    source_features = mixed_mel
  if source_features is None:
    raise ValueError('Either source_features or mixed_mel is required.')
  if interfering_features is None and interfering_mel is not None:
    interfering_features = interfering_mel
  if interfering_ids is None and interfering_text is not None:
    tok = tec_tokenizer.CharTokenizer(
        vocab_size=configs.TecBaseConfig.VOCAB_SIZE)
    texts = (
        [interfering_text]
        if isinstance(interfering_text, str) else list(interfering_text))
    interfering_ids, interfering_id_paddings = tok.batch_encode(texts)

  if isinstance(model_ref, (bytes, bytearray)):
    interpreter = tf.lite.Interpreter(model_content=bytes(model_ref))
  else:
    interpreter = tf.lite.Interpreter(model_path=str(model_ref))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  target_frames = 200
  target_bins = 128
  target_text_len = 100
  for detail in input_details:
    name = detail['name']
    shape = detail['shape']
    if 'source_features' in name and len(shape) == 3:
      target_frames = int(shape[1])
      target_bins = int(shape[2])
    elif 'interfering_ids' in name and len(shape) == 2:
      target_text_len = int(shape[1])

  src_feat, src_pad, orig_frames = _fit_3d_features(
      source_features, source_feature_paddings, target_frames, target_bins)

  if interfering_features is not None:
    int_feat, int_pad, _ = _fit_3d_features(
        interfering_features,
        interfering_feature_paddings,
        target_frames,
        target_bins,
        pad_value=-6.907755)
  else:
    int_feat = None
    int_pad = None

  if interfering_ids is not None:
    ids = np.asarray(interfering_ids, dtype=np.int32)
    if ids.ndim == 1:
      ids = ids[None, :]
    orig_t = int(ids.shape[1])
    if interfering_id_paddings is None:
      id_pad = np.zeros((ids.shape[0], orig_t), dtype=np.float32)
    else:
      id_pad = np.asarray(interfering_id_paddings, dtype=np.float32)
      if id_pad.ndim == 1:
        id_pad = id_pad[None, :]
    if orig_t > target_text_len:
      ids = ids[:, :target_text_len]
      id_pad = id_pad[:, :target_text_len]
    elif orig_t < target_text_len:
      extra_t = target_text_len - orig_t
      ids = np.pad(ids, ((0, 0), (0, extra_t)), constant_values=0)
      id_pad = np.pad(id_pad, ((0, 0), (0, extra_t)), constant_values=1.0)
  else:
    ids = None
    id_pad = None

  for detail in input_details:
    name = detail['name']
    if 'source_feature_paddings' in name:
      interpreter.set_tensor(detail['index'], src_pad)
    elif 'source_features' in name:
      interpreter.set_tensor(detail['index'], src_feat)
    elif 'interfering_feature_paddings' in name:
      val = (
          int_pad if int_pad is not None
          else np.zeros(detail['shape'], dtype=np.float32))
      interpreter.set_tensor(detail['index'], val)
    elif 'interfering_features' in name:
      val = (
          int_feat if int_feat is not None
          else np.full(detail['shape'], -6.907755, dtype=np.float32))
      interpreter.set_tensor(detail['index'], val)
    elif 'interfering_id_paddings' in name:
      val = (
          id_pad if id_pad is not None
          else np.zeros(detail['shape'], dtype=np.float32))
      interpreter.set_tensor(detail['index'], val)
    elif 'interfering_ids' in name:
      val = (
          ids if ids is not None
          else np.ones(detail['shape'], dtype=np.int32))
      interpreter.set_tensor(detail['index'], val)

  interpreter.invoke()

  outputs = {}
  for idx, detail in enumerate(interpreter.get_output_details()):
    tensor = interpreter.get_tensor(detail['index'])
    if tensor.shape[-1] == 1 or 'eos' in detail['name']:
      outputs['eos_probs'] = tensor
    else:
      outputs['feature_preds'] = tensor
    outputs[f'output_{idx}'] = tensor
  if 'feature_preds' in outputs:
    outputs['predicted_mel'] = outputs['feature_preds'][0, :orig_frames, :]
  return outputs
