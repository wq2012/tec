"""Tests for tec.waveform_processor."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tec import waveform_processor  # noqa: E402
from lingvo import compat as tf  # noqa: E402
from lingvo.core import test_utils  # noqa: E402


class WaveformProcessorTest(test_utils.TestCase):

  def testComputeFftSize(self):
    self.assertEqual(
        2048, waveform_processor.compute_fft_size(50.0, 24000.0))
    self.assertEqual(
        512, waveform_processor.compute_fft_size(25.0, 16000.0))

  def testSamplesAndFramesConversion(self):
    wp = waveform_processor.WaveformProcessor.Params().Instantiate()
    num_frames = wp.SamplesToFrames(4800)
    self.assertEqual(13, num_frames)
    self.assertEqual(4800, wp.FramesToSamples(num_frames))

  def testWaveformsToSpectrogramsAndGriffinLim(self):
    with self.session(use_gpu=False) as sess:
      p = waveform_processor.WaveformProcessor.Params()
      p.num_griffin_lim_iters = 4
      wp = p.Instantiate()

      batch_size = 2
      num_samples = 2400
      t = np.linspace(0.0, 0.1, num_samples, endpoint=False, dtype=np.float32)
      tone = 0.25 * np.sin(2.0 * np.pi * 440.0 * t)
      waveforms = tf.constant(np.stack([tone, 0.5 * tone], axis=0))
      paddings = tf.zeros([batch_size, num_samples], dtype=tf.float32)

      spec_out = wp.WaveformsToSpectrograms(waveforms, paddings)
      recon_wav, recon_pad = wp.SpectrogramsToWaveforms(
          spec_out.spectrograms, spec_out.paddings)

      mel_val, raw_val, pad_val, wav_val, wav_pad_val = sess.run([
          spec_out.spectrograms,
          spec_out.raw_spectrograms,
          spec_out.paddings,
          recon_wav,
          recon_pad,
      ])
      expected_frames = wp.SamplesToFrames(num_samples)
      self.assertEqual((batch_size, expected_frames, 128), mel_val.shape)
      self.assertEqual((batch_size, expected_frames, 1025), raw_val.shape)
      self.assertEqual((batch_size, expected_frames), pad_val.shape)
      self.assertEqual(batch_size, wav_val.shape[0])
      self.assertEqual(wav_val.shape, wav_pad_val.shape)


if __name__ == '__main__':
  tf.test.main()
