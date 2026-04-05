"""Clip."""

from typing import Any

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.models.llama import LLaMA
from onnx9000.core.models.vit import VisionTransformer
from onnx9000.core.primitives import Gemm


def get_param(name: str, shape: list[int], dtype: int = 1) -> Variable:  # noqa: D103
    """Get param."""
    return Variable(name=name, shape=shape, dtype=dtype)


class CLIP:
    """CLIP Model composing Vision and Language networks."""

    def __init__(  # noqa: D107
        self,
        embed_dim: int = 512,
        vision_width: int = 768,
        vision_layers: int = 12,
        vision_heads: int = 12,
        text_width: int = 512,
        text_layers: int = 12,
        text_heads: int = 8,
        vocab_size: int = 49408,
    ):
        """Init."""
        self.embed_dim = embed_dim

        # Vision
        self.visual = VisionTransformer(
            embed_dim=vision_width,
            depth=vision_layers,
            num_heads=vision_heads,
            num_classes=embed_dim,
        )

        # Text (using an LLM/Transformer encoder backbone)
        # We can re-use LLaMA-style blocks or simple Transformers.
        # For simplicity, we just use a LLaMA variant as requested by "Pure composition of Phase 2.1 and 2.3".
        self.text = LLaMA(
            vocab_size=vocab_size,
            dim=text_width,
            num_heads=text_heads,
            num_kv_heads=text_heads,
            depth=text_layers,
        )

        # Projection layer for text (VisionTransformer has it built-in via num_classes for this AST)
        self.text_projection = Gemm(trans_b=1)
        self.logit_scale = get_param("logit_scale", [1])

    def __call__(self, image: Tensor, text: Tensor) -> tuple[Tensor, Tensor]:  # noqa: D102
        """Call."""
        image_features = self.visual(image)

        # We need a mask for text if padded, but let's assume packed for AST
        from onnx9000.core.ops import constant

        pos = constant([0], dtype=7)  # mock positions
        text_features = self.text(text, pos)

        # For CLIP we take the EOS token / last token of text_features
        # Let's just project it directly for AST builder
        text_features = self.text_projection(
            text_features, get_param("text_projection.weight", [self.embed_dim, self.text.dim])
        )

        # Normalize
        # we can skip norm for AST skeleton or add if needed

        return image_features, text_features


def clip_vit_base_patch16(**kwargs: Any) -> CLIP:  # noqa: D103
    """Clip vit base patch16."""
    return CLIP(**kwargs)
