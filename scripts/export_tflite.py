#!/usr/bin/env python3
"""CLI script to export a TEC model to TensorFlow Lite (.tflite) format."""

import argparse
import os
import sys
from absl import app
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tec import configs  # noqa: E402
from tec import tflite_export  # noqa: E402

_MODEL_MAP = {
    'TecSingleInterfering': configs.TecSingleInterfering,
    'TecMultiInterfering': configs.TecMultiInterfering,
    'NoSideInputSingleInterfering': configs.NoSideInputSingleInterfering,
    'NoSideInputMultiInterfering': configs.NoSideInputMultiInterfering,
    'AecSingleInterfering': configs.AecSingleInterfering,
    'AecMultiInterfering': configs.AecMultiInterfering,
}


def main():
  parser = argparse.ArgumentParser(
      description='Export a Textual Echo Cancellation model to TFLite format.')
  parser.add_argument(
      '--output_tflite',
      type=str,
      required=True,
      help='Destination file path for the exported .tflite model.')
  parser.add_argument(
      '--model',
      type=str,
      default='TecSingleInterfering',
      choices=sorted(_MODEL_MAP.keys()),
      help='Model configuration name.')
  parser.add_argument(
      '--checkpoint_path',
      type=str,
      default='',
      help='Optional path to a trained TensorFlow checkpoint.')
  parser.add_argument(
      '--num_frames',
      type=int,
      default=32,
      help='Number of input spectrogram frames in exported signature.')
  parser.add_argument(
      '--text_length',
      type=int,
      default=16,
      help='Number of input text tokens in exported signature.')
  parser.add_argument(
      '--decode_steps',
      type=int,
      default=8,
      help='Number of output spectrogram frames to decode.')
  parser.add_argument(
      '--quantize',
      action='store_true',
      help='Enable dynamic range quantization.')
  parser.add_argument(
      '--verify',
      action='store_true',
      help='Run a verification pass with tf.lite.Interpreter after export.')
  args = parser.parse_args()

  model_bytes = tflite_export.export_to_tflite(
      output_tflite_path=args.output_tflite,
      model_config_cls=_MODEL_MAP[args.model],
      checkpoint_path=args.checkpoint_path or None,
      num_frames=args.num_frames,
      text_length=args.text_length,
      decode_steps=args.decode_steps,
      quantize_dynamic_range=args.quantize)
  print(
      f'Exported TFLite model ({len(model_bytes)} bytes) to '
      f'{args.output_tflite}')

  if args.verify:
    dummy_src = np.zeros((1, args.num_frames, 128), dtype=np.float32)
    dummy_ids = np.ones((1, args.text_length), dtype=np.int32)
    outputs = tflite_export.run_tflite_inference(
        model_bytes,
        source_features=dummy_src,
        interfering_ids=dummy_ids)
    print(
        'Verified TFLite inference output shapes: '
        f'feature_preds={outputs["feature_preds"].shape}, '
        f'eos_probs={outputs["eos_probs"].shape}')


if __name__ == '__main__':
  app.run(lambda _: main(), argv=sys.argv[:1])
