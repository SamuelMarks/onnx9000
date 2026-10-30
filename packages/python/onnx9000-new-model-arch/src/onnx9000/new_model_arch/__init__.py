"""Modern model architecture builders and registry for ONNX9000."""

from onnx9000.core.models import (
    CLIP,
    RWKV,
    DiT,
    EfficientNet,
    LLaMA,
    Mamba,
    MaskedAutoencoderViT,
    Mixtral,
    MobileViT,
    ResNet,
    SwinTransformer,
    VisionTransformer,
    Whisper,
    clip_vit_base_patch16,
    convnext_tiny,
    dit_xl_2,
    efficientnet_b0,
    llama_7b,
    mae_vit_base_patch16,
    mamba_130m,
    mistral_7b,
    mixtral_8x7b,
    mobilevit_s,
    resnet18,
    resnet50,
    rwkv_v4,
    swin_t,
    vit_base_patch16_224,
    whisper_tiny,
)

ARCHITECTURE_NAMES: tuple[str, ...] = (
    "clip",
    "convnext",
    "dit",
    "efficientnet",
    "llama",
    "mae",
    "mamba",
    "mistral",
    "mixtral",
    "mobilevit",
    "resnet18",
    "resnet50",
    "rwkv",
    "swin",
    "vit",
    "whisper",
)


class NewModelArch:
    """Registry and factory for modern ONNX9000 neural network architectures."""

    def list_architectures(self) -> list[str]:
        """Return all supported model architecture names.

        Returns:
            List of canonical architecture identifiers.
        """
        return list(ARCHITECTURE_NAMES)

    def process(self, arch_name: str) -> str:
        """Process a model architecture request by name.

        Args:
            arch_name: Name of the architecture.

        Returns:
            Formatted model architecture status string.
        """
        return f"New Model Arch processed {arch_name}"


__all__ = [
    "ARCHITECTURE_NAMES",
    "CLIP",
    "RWKV",
    "DiT",
    "EfficientNet",
    "LLaMA",
    "Mamba",
    "MaskedAutoencoderViT",
    "Mixtral",
    "MobileViT",
    "NewModelArch",
    "ResNet",
    "SwinTransformer",
    "VisionTransformer",
    "Whisper",
    "clip_vit_base_patch16",
    "convnext_tiny",
    "dit_xl_2",
    "efficientnet_b0",
    "llama_7b",
    "mae_vit_base_patch16",
    "mamba_130m",
    "mistral_7b",
    "mixtral_8x7b",
    "mobilevit_s",
    "resnet18",
    "resnet50",
    "rwkv_v4",
    "swin_t",
    "vit_base_patch16_224",
    "whisper_tiny",
]
