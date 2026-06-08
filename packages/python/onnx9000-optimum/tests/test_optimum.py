from onnx9000_optimum.architectures import (
    BARTConfig,
    BaseConfig,
    BERTConfig,
    CLIPConfig,
    DETRConfig,
    DistilBERTConfig,
    GemmaConfig,
    GPT2Config,
    LLaMAConfig,
    LlamaVisionConfig,
    MistralConfig,
    ORTModelForCausalLM,
    ORTModelForImageClassification,
    ORTModelForMaskedLM,
    ORTModelForObjectDetection,
    ORTModelForQuestionAnswering,
    ORTModelForSemanticSegmentation,
    ORTModelForSeq2SeqLM,
    ORTModelForSequenceClassification,
    ORTModelForSpeechSeq2Seq,
    ORTModelForTokenClassification,
    PhiConfig,
    QwenConfig,
    RoBERTaConfig,
    SpeechT5Config,
    StableDiffusionTextEncoderConfig,
    StableDiffusionUNetConfig,
    StableDiffusionVAEConfig,
    T5Config,
    ViTConfig,
    Wav2Vec2Config,
    WhisperConfig,
    YOLOSConfig,
)


def test_optimum_architectures():
    assert BaseConfig() is not None
    assert BERTConfig() is not None
    assert RoBERTaConfig() is not None
    assert DistilBERTConfig() is not None
    assert T5Config() is not None
    assert BARTConfig() is not None
    assert GPT2Config() is not None
    assert LLaMAConfig() is not None
    assert MistralConfig() is not None
    assert GemmaConfig() is not None
    assert PhiConfig() is not None
    assert QwenConfig() is not None
    assert LlamaVisionConfig() is not None
    assert ViTConfig() is not None
    assert CLIPConfig() is not None
    assert DETRConfig() is not None
    assert YOLOSConfig() is not None
    assert StableDiffusionUNetConfig() is not None
    assert StableDiffusionVAEConfig() is not None
    assert StableDiffusionTextEncoderConfig() is not None
    assert WhisperConfig() is not None
    assert Wav2Vec2Config() is not None
    assert SpeechT5Config() is not None
    assert ORTModelForSequenceClassification() is not None
    assert ORTModelForTokenClassification() is not None
    assert ORTModelForQuestionAnswering() is not None
    assert ORTModelForCausalLM() is not None
    assert ORTModelForMaskedLM() is not None
    assert ORTModelForSeq2SeqLM() is not None
    assert ORTModelForImageClassification() is not None
    assert ORTModelForObjectDetection() is not None
    assert ORTModelForSpeechSeq2Seq() is not None
    assert ORTModelForSemanticSegmentation() is not None
