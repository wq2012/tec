#!/usr/bin/env python3
"""CLI script to run TEC, AEC-Seq2seq, Vanilla-Seq2seq, or NLMS inference."""

import argparse
import os
import sys
from absl import app

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tec import aec_model  # noqa: E402
from tec import configs  # noqa: E402
from tec import data_prep  # noqa: E402
from tec import inference  # noqa: E402

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
      description='Run echo cancellation inference on a mixed WAV file.')
  parser.add_argument(
      '--mixed_wav',
      type=str,
      required=True,
      help='Path to the mixed (noisy/reverberant) input WAV file.')
  parser.add_argument(
      '--output_wav',
      type=str,
      required=True,
      help='Path to write the enhanced clean output WAV file.')
  parser.add_argument(
      '--interfering_text',
      type=str,
      default='',
      help='Interfering TTS transcript (side input for TEC).')
  parser.add_argument(
      '--interfering_wav',
      type=str,
      default='',
      help='Path to clean interfering reference WAV (for AEC-Seq2seq or NLMS).')
  parser.add_argument(
      '--model',
      type=str,
      default='TecSingleInterfering',
      choices=sorted(list(_MODEL_MAP.keys()) + ['NlmsAec']),
      help='Model type to run.')
  parser.add_argument(
      '--checkpoint_path',
      type=str,
      default='',
      help='Optional path to a trained TensorFlow model checkpoint.')
  parser.add_argument(
      '--decode_max_output_frames',
      type=int,
      default=200,
      help='Maximum number of spectrogram frames to decode.')
  args = parser.parse_args()

  mixed_wav, sr = data_prep.read_wav_file(args.mixed_wav)
  int_wav = None
  if args.interfering_wav:
    int_wav, _ = data_prep.read_wav_file(args.interfering_wav)

  if args.model == 'NlmsAec':
    if int_wav is None:
      raise ValueError('--interfering_wav is required when --model=NlmsAec')
    nlms = aec_model.NlmsAec()
    cleaned_wav, _, _ = nlms.process(mixed_wav, int_wav)
    data_prep.write_wav_file(args.output_wav, cleaned_wav, sample_rate=sr)
    print(f'Wrote NLMS-cleaned waveform to {args.output_wav}')
    return

  runner = inference.TecInferenceRunner(
      model_config_cls=_MODEL_MAP[args.model],
      checkpoint_path=args.checkpoint_path or None,
      decode_max_output_frames=args.decode_max_output_frames)
  try:
    outputs = runner.predict(
        mixed_waveforms=[mixed_wav],
        interfering_texts=[args.interfering_text],
        interfering_waveforms=[int_wav] if int_wav is not None else None)
    pred_wav = outputs['predicted_waveforms'][0]
    pred_len = int(outputs['predicted_waveform_lengths'][0])
    if pred_len > 0:
      pred_wav = pred_wav[:pred_len]
    data_prep.write_wav_file(args.output_wav, pred_wav, sample_rate=sr)
    print(f'Wrote enhanced waveform to {args.output_wav}')
  finally:
    runner.close()


if __name__ == '__main__':
  app.run(lambda _: main(), argv=sys.argv[:1])
