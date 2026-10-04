"""Tests for tec.data_prep and tec.evaluation."""

import os
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tec import data_prep  # noqa: E402
from tec import evaluation  # noqa: E402


class DataPrepAndEvaluationTest(unittest.TestCase):

  def testPairUtterancesRequiresStrictlyLongerInterfering(self):
    clean = [
        data_prep.UtteranceRecord('c1', np.zeros(100, dtype=np.float32), 'a'),
        data_prep.UtteranceRecord('c2', np.zeros(500, dtype=np.float32), 'b'),
    ]
    interfering = [
        data_prep.UtteranceRecord('i1', np.zeros(200, dtype=np.float32), 'x'),
        data_prep.UtteranceRecord('i2', np.zeros(300, dtype=np.float32), 'y'),
    ]
    pairs = data_prep.pair_utterances(clean, interfering, seed=42)
    self.assertEqual(1, len(pairs))
    self.assertEqual('c1', pairs[0][0].utt_id)
    self.assertGreater(len(pairs[0][1].waveform), len(pairs[0][0].waveform))

  def testReverberationAndSnrMixing(self):
    rir = data_prep.generate_synthetic_rir(
        sample_rate=24000, rt60=0.2, duration_sec=0.05, seed=1)
    self.assertAlmostEqual(1.0, float(np.linalg.norm(rir)), places=5)

    clean = np.ones(1000, dtype=np.float32) * 0.1
    interfering = np.ones(1500, dtype=np.float32) * 0.2
    reverb_int = data_prep.apply_reverberation(interfering, rir)
    self.assertEqual(1500, len(reverb_int))

    mixed, padded_clean = data_prep.mix_waveforms_at_snr(
        clean, reverb_int, snr_db=0.0)
    self.assertEqual(1500, len(mixed))
    self.assertEqual(1500, len(padded_clean))
    np.testing.assert_allclose(0.0, padded_clean[1000:])

    active_clean_power = np.mean(np.square(padded_clean[:1000]))
    scaled_int = mixed - padded_clean
    int_power = np.mean(np.square(scaled_int))
    measured_snr_db = 10.0 * np.log10(active_clean_power / int_power)
    self.assertAlmostEqual(0.0, measured_snr_db, places=4)

  def testWavReadWriteRoundTrip(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      wav_path = os.path.join(tmpdir, 'sample.wav')
      orig = np.sin(np.linspace(0, 6.28, 480, dtype=np.float32)) * 0.5
      data_prep.write_wav_file(wav_path, orig, sample_rate=24000)
      loaded, sr = data_prep.read_wav_file(wav_path)
      self.assertEqual(24000, sr)
      np.testing.assert_allclose(orig, loaded, atol=1e-4)

  def testMcdAndDtwEvaluation(self):
    rng = np.random.RandomState(7)
    ref_mel = rng.normal(0.0, 1.0, size=(20, 128)).astype(np.float32)
    mcd_zero = evaluation.compute_mcd(ref_mel, ref_mel)
    self.assertAlmostEqual(0.0, mcd_zero, places=5)

    noisy_mel = ref_mel + rng.normal(0.0, 0.5, size=(20, 128)).astype(
        np.float32)
    mcd_noisy = evaluation.compute_mcd(ref_mel, noisy_mel)
    self.assertGreater(mcd_noisy, 0.0)

  def testTranscriptNormalizationAndWer(self):
    norm = evaluation.normalize_transcript('Hello, World! How are you?')
    self.assertEqual('hello world how are you', norm)

    wer_exact = evaluation.compute_wer(
        ['Turn off the alarm!'], ['turn off the alarm'])
    self.assertAlmostEqual(0.0, wer_exact['wer'])

    wer_sub = evaluation.compute_wer(
        ['set alarm for seven am'], ['set alarm for eight am'])
    self.assertAlmostEqual(20.0, wer_sub['wer'])
    self.assertEqual(1, wer_sub['word_errors'])
    self.assertEqual(5, wer_sub['total_words'])

  def testFlopsAndSideInputEstimates(self):
    estimates = evaluation.estimate_flops_and_side_input(5.0)
    self.assertAlmostEqual(2.1e9, estimates['Vanilla-Seq2seq']['flops'])
    self.assertAlmostEqual(2.5e9, estimates['AEC-Seq2seq']['flops'])
    self.assertLess(
        estimates['TEC']['flops'], estimates['AEC-Seq2seq']['flops'])
    self.assertLess(estimates['TEC']['side_input_bytes'], 1024.0)
    self.assertEqual(240000.0, estimates['AEC-Seq2seq']['side_input_bytes'])

  def testResamplingAndShorterInterferingPadding(self):
    orig = np.sin(np.linspace(0, 6.28, 2205, dtype=np.float32)) * 0.4
    resampled = data_prep.resample_waveform(orig, 22050, 24000)
    self.assertEqual(2400, len(resampled))

    clean = np.ones(1200, dtype=np.float32) * 0.1
    short_int = np.ones(600, dtype=np.float32) * 0.2
    mixed, padded_clean = data_prep.mix_waveforms_at_snr(
        clean, short_int, snr_db=0.0)
    self.assertEqual(1200, len(mixed))
    self.assertEqual(1200, len(padded_clean))


if __name__ == '__main__':
  unittest.main()
