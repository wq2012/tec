"""TFLite export and on-device inference for Textual Echo Cancellation."""

import os
from typing import Dict, Optional
from lingvo import compat as tf
from lingvo.core import py_utils
import numpy as np
from tec import configs


def export_to_tflite(
    output_tflite_path: str,
    model_config_cls=configs.TecSingleInterfering,
    checkpoint_path: Optional[str] = None,
    num_frames: int = 32,
    text_length: int = 16,
    decode_steps: int = 8,
    quantize_dynamic_range: bool = False,
) -> bytes:
  """Exports a TEC / AEC / Vanilla-Seq2seq model to a TFLite FlatBuffer."""
  graph = tf.Graph()
  with graph.as_default():
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
            interfering_features=source_features,
            interfering_feature_paddings=source_feature_paddings),
        tgt=py_utils.NestedMap(
            features=tf.zeros(
                [1, decode_steps, num_mel_bins], dtype=tf.float32),
            feature_paddings=tf.zeros([1, decode_steps], dtype=tf.float32)))

    predictions = task.ComputePredictions(task.theta, input_batch)
    feature_preds = tf.identity(
        predictions.feature_preds, name='feature_preds')
    eos_probs = tf.identity(predictions.eos_probs, name='eos_probs')

    input_tensors = [
        source_features,
        source_feature_paddings,
        interfering_ids,
        interfering_id_paddings,
    ]
    if task_p.encoder is None:
      input_tensors = [source_features, source_feature_paddings]
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


def run_tflite_inference(
    tflite_model_or_path,
    source_features: np.ndarray,
    source_feature_paddings: Optional[np.ndarray] = None,
    interfering_ids: Optional[np.ndarray] = None,
    interfering_id_paddings: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
  """Executes a converted TEC `.tflite` model using `tf.lite.Interpreter`."""
  if isinstance(tflite_model_or_path, (bytes, bytearray)):
    interpreter = tf.lite.Interpreter(model_content=bytes(tflite_model_or_path))
  else:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_or_path))
  interpreter.allocate_tensors()

  src_feat = np.asarray(source_features, dtype=np.float32)
  if source_feature_paddings is None:
    src_pad = np.zeros(src_feat.shape[:2], dtype=np.float32)
  else:
    src_pad = np.asarray(source_feature_paddings, dtype=np.float32)

  for detail in interpreter.get_input_details():
    name = detail['name']
    if 'source_feature_paddings' in name:
      interpreter.set_tensor(detail['index'], src_pad)
    elif 'source_features' in name:
      interpreter.set_tensor(detail['index'], src_feat)
    elif 'interfering_id_paddings' in name:
      if interfering_id_paddings is None:
        id_pad = np.zeros(detail['shape'], dtype=np.float32)
      else:
        id_pad = np.asarray(interfering_id_paddings, dtype=np.float32)
      interpreter.set_tensor(detail['index'], id_pad)
    elif 'interfering_ids' in name:
      if interfering_ids is None:
        ids = np.ones(detail['shape'], dtype=np.int32)
      else:
        ids = np.asarray(interfering_ids, dtype=np.int32)
      interpreter.set_tensor(detail['index'], ids)

  interpreter.invoke()

  outputs = {}
  for idx, detail in enumerate(interpreter.get_output_details()):
    tensor = interpreter.get_tensor(detail['index'])
    if tensor.shape[-1] == 1 or 'eos' in detail['name']:
      outputs['eos_probs'] = tensor
    else:
      outputs['feature_preds'] = tensor
    outputs[f'output_{idx}'] = tensor
  return outputs
