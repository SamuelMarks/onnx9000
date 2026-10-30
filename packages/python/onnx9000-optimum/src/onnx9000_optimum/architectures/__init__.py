"""ONNX configuration and model wrappers for specific transformer and vision architectures."""

from typing import TypeAlias

ConfigArgValue: TypeAlias = str | int | float | bool | list[str] | list[int] | None


class BaseConfig:
    """Base configuration class for Optimum architecture exports."""

    def __init__(self, **kwargs: ConfigArgValue) -> None:
        """Initialize configuration with optional architecture parameters.

        Args:
            **kwargs: Arbitrary configuration key-value parameters.
        """
        self.options: dict[str, ConfigArgValue] = dict(kwargs)


class BERTConfig(BaseConfig):
    """ONNX export configuration for BERT architectures."""


class RoBERTaConfig(BaseConfig):
    """ONNX export configuration for RoBERTa architectures."""


class DistilBERTConfig(BaseConfig):
    """ONNX export configuration for DistilBERT architectures."""


class T5Config(BaseConfig):
    """ONNX export configuration for T5 encoder-decoder architectures."""


class BARTConfig(BaseConfig):
    """ONNX export configuration for BART encoder-decoder architectures."""


class GPT2Config(BaseConfig):
    """ONNX export configuration for GPT-2 causal language models."""


class LLaMAConfig(BaseConfig):
    """ONNX export configuration for LLaMA causal language models."""


class MistralConfig(BaseConfig):
    """ONNX export configuration for Mistral causal language models."""


class GemmaConfig(BaseConfig):
    """ONNX export configuration for Gemma causal language models."""


class PhiConfig(BaseConfig):
    """ONNX export configuration for Phi causal language models."""


class QwenConfig(BaseConfig):
    """ONNX export configuration for Qwen causal language models."""


class LlamaVisionConfig(BaseConfig):
    """ONNX export configuration for LLaMA-Vision multimodal models."""


class ViTConfig(BaseConfig):
    """ONNX export configuration for Vision Transformer (ViT) models."""


class CLIPConfig(BaseConfig):
    """ONNX export configuration for CLIP vision-language models."""


class DETRConfig(BaseConfig):
    """ONNX export configuration for DETR object detection models."""


class YOLOSConfig(BaseConfig):
    """ONNX export configuration for YOLOS object detection models."""


class StableDiffusionUNetConfig(BaseConfig):
    """ONNX export configuration for Stable Diffusion UNet denoiser models."""


class StableDiffusionVAEConfig(BaseConfig):
    """ONNX export configuration for Stable Diffusion VAE encoder/decoder models."""


class StableDiffusionTextEncoderConfig(BaseConfig):
    """ONNX export configuration for Stable Diffusion text encoder models."""


class WhisperConfig(BaseConfig):
    """ONNX export configuration for Whisper speech recognition models."""


class Wav2Vec2Config(BaseConfig):
    """ONNX export configuration for Wav2Vec2 speech models."""


class SpeechT5Config(BaseConfig):
    """ONNX export configuration for SpeechT5 audio synthesis/recognition models."""


class ORTModelForSequenceClassification(BaseConfig):
    """Optimum-compatible ONNX Runtime wrapper for sequence classification."""


class ORTModelForTokenClassification(BaseConfig):
    """Optimum-compatible ONNX Runtime wrapper for token classification."""


class ORTModelForQuestionAnswering(BaseConfig):
    """Optimum-compatible ONNX Runtime wrapper for extractive question answering."""


class ORTModelForCausalLM(BaseConfig):
    """Optimum-compatible ONNX Runtime wrapper for causal language modeling."""


class ORTModelForMaskedLM(BaseConfig):
    """Optimum-compatible ONNX Runtime wrapper for masked language modeling."""


class ORTModelForSeq2SeqLM(BaseConfig):
    """Optimum-compatible ONNX Runtime wrapper for sequence-to-sequence language modeling."""


class ORTModelForImageClassification(BaseConfig):
    """Optimum-compatible ONNX Runtime wrapper for image classification."""


class ORTModelForObjectDetection(BaseConfig):
    """Optimum-compatible ONNX Runtime wrapper for object detection."""


class ORTModelForSpeechSeq2Seq(BaseConfig):
    """Optimum-compatible ONNX Runtime wrapper for speech sequence-to-sequence models."""


class ORTModelForSemanticSegmentation(BaseConfig):
    """Optimum-compatible ONNX Runtime wrapper for semantic segmentation."""
