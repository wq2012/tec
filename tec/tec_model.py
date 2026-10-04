"""Textual Echo Cancellation (TEC) and Vanilla-Seq2seq models."""

from lingvo import compat as tf
from lingvo.core import base_model
from lingvo.core import metrics
from lingvo.core import py_utils
import numpy as np
from tec import decoder as tec_decoder
from tec import encoder_speech
from tec import encoder_text
from tec import waveform_processor


class TecModel(base_model.BaseTask):
  """Multi-source sequence-to-sequence model for Textual Echo Cancellation.

  Encodes the noisy/reverberant microphone speech (`encoder_speech`) and the
  interfering TTS transcript (`encoder`), and decodes the enhanced user speech
  log-Mel spectrogram using `MultiSourceFbeDecoderV1`. When `encoder` is None,
  reduces to the single-source `VanillaSeq2SeqModel` baseline.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'tec_model'
    p.encoder = encoder_text.TtsEncoderV2.Params()
    p.decoder = tec_decoder.MultiSourceFbeDecoderV1.Params()
    p.Define('encoder_speech', encoder_speech.SpeechEncoderV1.Params(),
             'Audio encoder parameters for the microphone mixture.')
    p.Define('waveform_processor',
             waveform_processor.WaveformProcessor.Params(),
             'Frontend parameters for spectrogram-to-waveform synthesis.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    with tf.variable_scope(p.name, reuse=tf.AUTO_REUSE):
      if p.encoder is not None:
        self.CreateChild('encoder', p.encoder)
      self.CreateChild('encoder_speech', p.encoder_speech)
      self.CreateChild('decoder', p.decoder)
      if p.waveform_processor:
        self.CreateChild('waveform_processor', p.waveform_processor)

  def _Encode(self, theta, input_batch):
    """Encodes microphone speech (`source_0`) and optional text (`source_1`)."""
    p = self.params
    speech_in = py_utils.NestedMap(
        src_inputs=input_batch.src.source_features,
        paddings=input_batch.src.source_feature_paddings)
    speech_enc = self.encoder_speech.FProp(theta.encoder_speech, speech_in)

    if p.encoder is None:
      return speech_enc

    text_enc = self.encoder.FProp(theta.encoder, input_batch.src)
    return py_utils.NestedMap(
        encoded=py_utils.NestedMap(
            source_0=speech_enc.encoded, source_1=text_enc.encoded),
        padding=py_utils.NestedMap(
            source_0=speech_enc.padding, source_1=text_enc.padding))

  def ComputePredictions(self, theta, input_batch):
    """Computes teacher-forced spectrogram predictions."""
    encoder_outputs = self._Encode(theta, input_batch)
    predictions = self.decoder.ComputePredictions(
        theta.decoder, encoder_outputs, input_batch.tgt)
    predictions.encoder_outputs = encoder_outputs
    return predictions

  def ComputeLoss(self, theta, predictions, input_batch):
    """Computes L1 + L2 spectrogram losses and stop-token BCE loss."""
    return self.decoder.ComputeLoss(
        theta.decoder, predictions, input_batch.tgt)

  def Decode(self, input_batch):
    """Runs autoregressive inference and Griffin-Lim waveform synthesis."""
    p = self.params
    theta = self.theta
    with tf.name_scope('decode'):
      encoder_outputs = self._Encode(theta, input_batch)
      decoded = self.decoder.Decode(encoder_outputs, input_batch.tgt)

      out = py_utils.NestedMap(
          utt_id=input_batch.utt_id,
          feature_preds=decoded.feature_preds,
          feature_preds_pre=decoded.feature_preds_pre,
          feature_paddings=decoded.paddings,
          eos_probs=decoded.eos_probs,
          attention=decoded.attention)

      if p.waveform_processor:
        pred_wav, pred_pad = self.waveform_processor.SpectrogramsToWaveforms(
            decoded.feature_preds, decoded.paddings)
        out.predicted_waveforms = pred_wav
        out.predicted_waveform_paddings = pred_pad
        out.predicted_waveform_lengths = tf.cast(
            tf.reduce_sum(1.0 - pred_pad, axis=1), tf.int32)
      return out

  def CreateDecoderMetrics(self):
    return {'num_samples_in_batch': metrics.AverageMetric()}

  def PostProcessDecodeOut(self, dec_out_dict, dec_metrics_dict):
    utt_ids = dec_out_dict['utt_id']
    batch_size = len(utt_ids)
    dec_metrics_dict['num_samples_in_batch'].Update(batch_size)
    outputs = []
    for idx in range(batch_size):
      uid = utt_ids[idx, 0]
      if isinstance(uid, bytes):
        uid = uid.decode('utf-8', errors='replace')
      outputs.append((uid, np.asarray(dec_out_dict['feature_preds'][idx])))
    return outputs


class VanillaSeq2SeqModel(TecModel):
  """Single-source Vanilla-Seq2seq baseline without side input."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = 'vanilla_seq2seq_model'
    p.encoder = None
    p.decoder = tec_decoder.FbeDecoderV1.Params()
    p.encoder_speech.num_conv_lstm_layers = 0
    return p
