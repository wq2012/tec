"""Input generator for Textual Echo Cancellation training and evaluation."""

import glob
from lingvo import compat as tf
from lingvo.core import base_input_generator
from lingvo.core import py_utils
from tec import waveform_processor


class TecInputGenerator(base_input_generator.BaseInputGenerator):
  """Generates batches of mixed speech, interfering inputs, and clean speech."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'tec_input_generator'
    p.Define('file_pattern', '',
             'Glob pattern of TFRecord files containing prepared utterances.')
    p.batch_size = 4
    p.Define('vocab_size', 96, 'Character vocabulary size.')
    p.Define('waveform_processor',
             waveform_processor.WaveformProcessor.Params(),
             'WaveformProcessor params for spectrogram extraction.')
    p.Define('shuffle', True, 'Whether to shuffle examples during training.')
    p.Define('shuffle_buffer_size', 256,
             'Buffer size for TFRecord dataset shuffling.')
    p.Define('repeat', True, 'Whether to repeat the dataset indefinitely.')
    p.Define('use_synthetic_data', False,
             'If True (or if file_pattern is empty), generate synthetic data.')
    p.Define('synthetic_num_samples', 4800,
             'Number of waveform samples per utterance in synthetic mode.')
    p.Define('synthetic_text_len', 12,
             'Number of text tokens per utterance in synthetic mode.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self.CreateChild('waveform_processor', p.waveform_processor)
    self._next_element = None
    if p.file_pattern and not p.use_synthetic_data:
      self._build_tfrecord_pipeline()

  def _parse_tf_example(self, serialized):
    feature_spec = {
        'utt_id': tf.io.FixedLenFeature([], tf.string, default_value='utt_0'),
        'clean_waveform': tf.io.VarLenFeature(tf.float32),
        'interfering_waveform': tf.io.VarLenFeature(tf.float32),
        'mixed_waveform': tf.io.VarLenFeature(tf.float32),
        'clean_transcript': tf.io.FixedLenFeature(
            [], tf.string, default_value=''),
        'interfering_transcript': tf.io.FixedLenFeature(
            [], tf.string, default_value=''),
        'interfering_ids': tf.io.VarLenFeature(tf.int64),
    }
    parsed = tf.io.parse_single_example(serialized, feature_spec)
    clean_wav = tf.sparse.to_dense(parsed['clean_waveform'])
    interfering_wav = tf.sparse.to_dense(parsed['interfering_waveform'])
    mixed_wav = tf.sparse.to_dense(parsed['mixed_waveform'])
    interfering_ids = tf.cast(
        tf.sparse.to_dense(parsed['interfering_ids']), tf.int32)

    return {
        'utt_id': parsed['utt_id'],
        'clean_waveform': clean_wav,
        'clean_waveform_padding': tf.zeros_like(clean_wav),
        'interfering_waveform': interfering_wav,
        'interfering_waveform_padding': tf.zeros_like(interfering_wav),
        'mixed_waveform': mixed_wav,
        'mixed_waveform_padding': tf.zeros_like(mixed_wav),
        'clean_transcript': parsed['clean_transcript'],
        'interfering_transcript': parsed['interfering_transcript'],
        'interfering_ids': interfering_ids,
        'interfering_id_padding': tf.zeros_like(
            interfering_ids, dtype=tf.float32),
    }

  def _build_tfrecord_pipeline(self):
    p = self.params
    matched_files = sorted(glob.glob(p.file_pattern))
    if not matched_files:
      raise ValueError(f'No files matched file_pattern: {p.file_pattern}')
    files = tf.data.Dataset.from_tensor_slices(matched_files)
    if p.shuffle and len(matched_files) > 1:
      files = files.shuffle(buffer_size=len(matched_files))
    ds = files.interleave(
        tf.data.TFRecordDataset,
        cycle_length=min(4, len(matched_files)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if p.shuffle:
      ds = ds.shuffle(buffer_size=p.shuffle_buffer_size)
    if p.repeat:
      ds = ds.repeat()
    ds = ds.map(
        self._parse_tf_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    padded_shapes = {
        'utt_id': [],
        'clean_waveform': [None],
        'clean_waveform_padding': [None],
        'interfering_waveform': [None],
        'interfering_waveform_padding': [None],
        'mixed_waveform': [None],
        'mixed_waveform_padding': [None],
        'clean_transcript': [],
        'interfering_transcript': [],
        'interfering_ids': [None],
        'interfering_id_padding': [None],
    }
    padding_values = {
        'utt_id': '',
        'clean_waveform': 0.0,
        'clean_waveform_padding': 1.0,
        'interfering_waveform': 0.0,
        'interfering_waveform_padding': 1.0,
        'mixed_waveform': 0.0,
        'mixed_waveform_padding': 1.0,
        'clean_transcript': '',
        'interfering_transcript': '',
        'interfering_ids': 0,
        'interfering_id_padding': 1.0,
    }
    ds = ds.padded_batch(
        p.batch_size,
        padded_shapes=padded_shapes,
        padding_values=padding_values,
        drop_remainder=False)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
    self._next_element = iterator.get_next()

  def _waveforms_to_batch(self, raw):
    clean_spec = self.waveform_processor.WaveformsToSpectrograms(
        raw['clean_waveform'], raw['clean_waveform_padding'])
    mixed_spec = self.waveform_processor.WaveformsToSpectrograms(
        raw['mixed_waveform'], raw['mixed_waveform_padding'])
    interfering_spec = self.waveform_processor.WaveformsToSpectrograms(
        raw['interfering_waveform'], raw['interfering_waveform_padding'])

    src = py_utils.NestedMap(
        ids=raw['interfering_ids'],
        paddings=raw['interfering_id_padding'],
        transcripts=raw['interfering_transcript'],
        source_features=mixed_spec.spectrograms,
        source_feature_paddings=mixed_spec.paddings,
        source_waveforms=raw['mixed_waveform'],
        source_waveform_paddings=raw['mixed_waveform_padding'],
        interfering_features=interfering_spec.spectrograms,
        interfering_feature_paddings=interfering_spec.paddings,
        interfering_waveforms=raw['interfering_waveform'],
        interfering_waveform_paddings=raw['interfering_waveform_padding'])

    tgt = py_utils.NestedMap(
        features=clean_spec.spectrograms,
        feature_paddings=clean_spec.paddings,
        raw_spectrograms=clean_spec.raw_spectrograms,
        waveforms=raw['clean_waveform'],
        waveform_paddings=raw['clean_waveform_padding'],
        transcripts=raw['clean_transcript'])

    utt_id = tf.expand_dims(raw['utt_id'], -1)
    return py_utils.NestedMap(utt_id=utt_id, src=src, tgt=tgt)

  def _generate_synthetic_raw(self):
    p = self.params
    b = p.batch_size
    n = p.synthetic_num_samples
    t_len = p.synthetic_text_len
    seed = p.random_seed or 12345

    clean_wav = tf.random.normal([b, n], stddev=0.1, seed=seed)
    interfering_wav = tf.random.normal([b, n], stddev=0.1, seed=seed + 1)
    mixed_wav = clean_wav + interfering_wav
    wav_pad = tf.zeros([b, n], dtype=tf.float32)

    ids = tf.random.uniform(
        [b, t_len],
        minval=1,
        maxval=p.vocab_size,
        dtype=tf.int32,
        seed=seed + 2)
    id_pad = tf.zeros([b, t_len], dtype=tf.float32)

    return {
        'utt_id': tf.fill([b], 'synthetic_utt'),
        'clean_waveform': clean_wav,
        'clean_waveform_padding': wav_pad,
        'interfering_waveform': interfering_wav,
        'interfering_waveform_padding': wav_pad,
        'mixed_waveform': mixed_wav,
        'mixed_waveform_padding': wav_pad,
        'clean_transcript': tf.fill([b], 'turn on the lights'),
        'interfering_transcript': tf.fill([b], 'the weather is sunny today'),
        'interfering_ids': ids,
        'interfering_id_padding': id_pad,
    }

  def _InputBatch(self):
    if self._next_element is not None:
      return self._waveforms_to_batch(self._next_element)
    return self._waveforms_to_batch(self._generate_synthetic_raw())
