"""Textual Echo Cancellation (TEC) open-source library."""

from . import data_prep
from . import evaluation
from . import tokenizer

__version__ = '0.1.0'

UtteranceRecord = data_prep.UtteranceRecord
resample_waveform = data_prep.resample_waveform
pair_utterances = data_prep.pair_utterances
generate_synthetic_rir = data_prep.generate_synthetic_rir
apply_reverberation = data_prep.apply_reverberation
mix_waveforms_at_snr = data_prep.mix_waveforms_at_snr
pad_clean_to_match_mixed = data_prep.pad_clean_to_match_mixed
create_tf_example = data_prep.create_tf_example
prepare_tfrecord_dataset = data_prep.prepare_tfrecord_dataset

compute_mfcc_from_log_mel = evaluation.compute_mfcc_from_log_mel
compute_dtw_distance = evaluation.compute_dtw_distance
compute_mcd = evaluation.compute_mcd
normalize_transcript = evaluation.normalize_transcript
compute_wer = evaluation.compute_wer
estimate_flops_and_side_input = evaluation.estimate_flops_and_side_input

CharTokenizer = tokenizer.CharTokenizer

try:
  from . import aec_model
  from . import configs
  from . import decoder
  from . import encoder_speech
  from . import encoder_text
  from . import inference
  from . import input_generator
  from . import layers
  from . import multi_source_attention_steps
  from . import tec_model
  from . import tflite_export
  from . import waveform_processor

  AecModel = aec_model.AecModel
  NlmsAec = aec_model.NlmsAec

  TecBaseConfig = configs.TecBaseConfig
  TecSingleInterfering = configs.TecSingleInterfering
  TecMultiInterfering = configs.TecMultiInterfering
  NoSideInputSingleInterfering = configs.NoSideInputSingleInterfering
  NoSideInputMultiInterfering = configs.NoSideInputMultiInterfering
  AecSingleInterfering = configs.AecSingleInterfering
  AecMultiInterfering = configs.AecMultiInterfering

  AttentiveFbeDecoderStep = decoder.AttentiveFbeDecoderStep
  MultiSourceFbeDecoderV1Step = decoder.MultiSourceFbeDecoderV1Step
  FbeDecoderV1 = decoder.FbeDecoderV1
  MultiSourceFbeDecoderV1 = decoder.MultiSourceFbeDecoderV1

  SpeechEncoderV1 = encoder_speech.SpeechEncoderV1
  TtsEncoderV2 = encoder_text.TtsEncoderV2

  TecInferenceRunner = inference.TecInferenceRunner
  TecInputGenerator = input_generator.TecInputGenerator

  PostEditConvNet = layers.PostEditConvNet

  MultiSourceAttentionStep = (
      multi_source_attention_steps.MultiSourceAttentionStep)

  TecModel = tec_model.TecModel
  VanillaSeq2SeqModel = tec_model.VanillaSeq2SeqModel

  export_to_tflite = tflite_export.export_to_tflite
  run_tflite_inference = tflite_export.run_tflite_inference

  WaveformProcessor = waveform_processor.WaveformProcessor
  compute_fft_size = waveform_processor.compute_fft_size
except ImportError:
  pass
