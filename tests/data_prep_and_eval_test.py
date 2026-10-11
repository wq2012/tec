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
    # Direct path energy at direct_idx should exceed reverberant tail energy
    # (positive Direct-to-Reverberant Ratio).
    direct_idx = int(np.argmax(np.abs(rir)))
    direct_energy = float(rir[direct_idx] ** 2)
    tail_energy = max(1e-12, float(np.sum(rir ** 2) - direct_energy))
    drr_db = 10.0 * np.log10(direct_energy / tail_energy)
    self.assertGreater(drr_db, 3.0)

    clean = np.ones(1000, dtype=np.float32) * 0.1
    interfering = np.ones(1500, dtype=np.float32) * 0.2
    reverb_int = data_prep.apply_reverberation(interfering, rir)
    self.assertEqual(1500, len(reverb_int))

    mixed, padded_clean = data_prep.mix_waveforms_at_snr(
        clean, reverb_int, snr_db=0.0)
    self.assertEqual(1500, len(mixed))
    self.assertEqual(1500, len(padded_clean))
    np.testing.assert_allclose(0.0, padded_clean[1000:])

    active_clean_power = data_prep.compute_active_speech_power(clean)
    scaled_int = mixed - padded_clean
    int_power = data_prep.compute_active_speech_power(scaled_int)
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
    mfcc = evaluation.compute_mfcc_from_log_mel(ref_mel)
    self.assertEqual((20, 14), mfcc.shape)

    mcd_zero = evaluation.compute_mcd(ref_mel, ref_mel)
    self.assertAlmostEqual(0.0, mcd_zero, places=5)

    noisy_mel = ref_mel + rng.normal(0.0, 0.5, size=(20, 128)).astype(
        np.float32)
    mcd_noisy = evaluation.compute_mcd(ref_mel, noisy_mel)
    self.assertGreater(mcd_noisy, 0.0)

    # Verify Eq. 13 dB scaling constant (10 * sqrt(2) / ln(10)) and 13 MFCCs.
    raw_dist, _ = evaluation.compute_dtw_distance(
        mfcc[:, 1:14], mfcc[:, 1:14] + 1.0)
    expected_single_step = np.sqrt(13.0)
    self.assertAlmostEqual(20.0 * expected_single_step, raw_dist, places=4)

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
    single = evaluation.estimate_flops_and_side_input('single')
    self.assertGreater(single['AEC-Seq2seq']['flops'], single['TEC']['flops'])
    self.assertGreater(
        single['TEC']['flops'], single['Vanilla-Seq2seq']['flops'])
    self.assertAlmostEqual(0.10, single['TEC']['side_input_kb'])
    self.assertAlmostEqual(310.0, single['AEC-Seq2seq']['side_input_kb'])

    multi = evaluation.estimate_flops_and_side_input('multi')
    self.assertGreater(multi['AEC-Seq2seq']['flops'], multi['TEC']['flops'])
    self.assertGreater(
        multi['TEC']['flops'], multi['Vanilla-Seq2seq']['flops'])
    self.assertLess(multi['TEC']['flops'], single['TEC']['flops'])
    self.assertAlmostEqual(0.06, multi['TEC']['side_input_kb'])
    self.assertAlmostEqual(230.0, multi['AEC-Seq2seq']['side_input_kb'])

  def testBootstrapCiAndPairedSignificance(self):
    vals_a = [8.0, 8.5, 9.0, 8.2, 8.8, 9.1, 8.4, 8.6]
    vals_b = [6.0, 6.4, 6.9, 6.1, 6.7, 7.0, 6.3, 6.5]
    ci_a = evaluation.compute_bootstrap_ci(vals_a, num_resamples=1000, seed=1)
    self.assertLess(ci_a['ci_lower'], ci_a['mean'])
    self.assertGreater(ci_a['ci_upper'], ci_a['mean'])

    sig = evaluation.compute_paired_significance(
        vals_a, vals_b, num_resamples=1000, seed=1)
    self.assertGreater(sig['mean_diff'], 1.5)
    self.assertLess(sig['p_value_bootstrap'], 0.01)

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

    ex = data_prep.create_tf_example(
        utt_id='u1',
        clean_waveform=padded_clean,
        interfering_waveform=short_int,
        mixed_waveform=mixed,
        clean_transcript='hello',
        interfering_transcript='world',
        clean_num_samples=1000,
        interfering_num_samples=600)
    feat = ex.features.feature
    self.assertEqual(1000, feat['clean_num_samples'].int64_list.value[0])
    self.assertEqual(600, feat['interfering_num_samples'].int64_list.value[0])


if __name__ == '__main__':
  unittest.main()
