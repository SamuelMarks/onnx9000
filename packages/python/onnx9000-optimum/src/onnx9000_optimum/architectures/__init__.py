"""ONNX Config Mapping for specific Architectures."""

from typing import Any


class BaseConfig:
    """Configuration for BaseConfig."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()


class BERTConfig(BaseConfig):
    """Configuration for BERTConfig."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()


class RoBERTaConfig(BaseConfig):
    """Configuration for RoBERTaConfig."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class DistilBERTConfig(BaseConfig):
    """Configuration for DistilBERTConfig."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class T5Config(BaseConfig):
    """Configuration for T5Config."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class BARTConfig(BaseConfig):
    """Configuration for BARTConfig."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class GPT2Config(BaseConfig):
    """Configuration for GPT2Config."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class LLaMAConfig(BaseConfig):
    """Configuration for LLaMAConfig."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class MistralConfig(BaseConfig):
    """Configuration for MistralConfig."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class GemmaConfig(BaseConfig):
    """Configuration for GemmaConfig."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class PhiConfig(BaseConfig):
    """Configuration for PhiConfig."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class QwenConfig(BaseConfig):
    """Configuration for QwenConfig."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class LlamaVisionConfig(BaseConfig):
    """Configuration for LlamaVisionConfig."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class ViTConfig(BaseConfig):
    """Configuration for ViTConfig."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class CLIPConfig(BaseConfig):
    """Configuration for CLIPConfig."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class DETRConfig(BaseConfig):
    """Configuration for DETRConfig."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class YOLOSConfig(BaseConfig):
    """Configuration for YOLOSConfig."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class StableDiffusionUNetConfig(BaseConfig):
    """Configuration for StableDiffusionUNetConfig."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class StableDiffusionVAEConfig(BaseConfig):
    """Configuration for StableDiffusionVAEConfig."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class StableDiffusionTextEncoderConfig(BaseConfig):
    """Configuration for StableDiffusionTextEncoderConfig."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class WhisperConfig(BaseConfig):
    """Configuration for WhisperConfig."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class Wav2Vec2Config(BaseConfig):
    """Configuration for Wav2Vec2Config."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class SpeechT5Config(BaseConfig):
    """Configuration for SpeechT5Config."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class ORTModelForSequenceClassification:
    """Configuration for ORTModelForSequenceClassification."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class ORTModelForTokenClassification:
    """Configuration for ORTModelForTokenClassification."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class ORTModelForQuestionAnswering:
    """Configuration for ORTModelForQuestionAnswering."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class ORTModelForCausalLM:
    """Configuration for ORTModelForCausalLM."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class ORTModelForMaskedLM:
    """Configuration for ORTModelForMaskedLM."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class ORTModelForSeq2SeqLM:
    """Configuration for ORTModelForSeq2SeqLM."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class ORTModelForImageClassification:
    """Configuration for ORTModelForImageClassification."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class ORTModelForObjectDetection:
    """Configuration for ORTModelForObjectDetection."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class ORTModelForSpeechSeq2Seq:
    """Configuration for ORTModelForSpeechSeq2Seq."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover


class ORTModelForSemanticSegmentation:
    """Configuration for ORTModelForSemanticSegmentation."""

    def __init__(self, **kwargs: "Any") -> None:
        """Initialize."""
        super().__init__()  # pragma: no cover
