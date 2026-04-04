"""Module docstring."""

from typing import Any

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.ops import add, concat, flatten, transpose
from onnx9000.core.ops import slice as slice_op
from onnx9000.core.primitives import (
    ConvND,
    Gelu,
    Gemm,
    LayerNormalization,
    MultiHeadAttention,
)


def get_param(name: str, shape: list[int], dtype: int = 1) -> Variable:  # noqa: D103
    return Variable(name=name, shape=shape, dtype=dtype)


class PatchEmbed:
    """Image to Patch Embedding."""

    def __init__(  # noqa: D107
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        prefix: str = "",
    ):
        self.prefix = prefix
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = ConvND(2, in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def __call__(self, x: Tensor) -> Tensor:  # noqa: D102
        x = self.proj(
            x,
            get_param(
                f"{self.prefix}.proj.weight",
                [self.proj.out_channels, self.proj.in_channels, self.patch_size, self.patch_size],
            ),
            get_param(f"{self.prefix}.proj.bias", [self.proj.out_channels]),
        )
        # flatten: [B, C, H, W] -> [B, C, H*W]
        from onnx9000.core.ops import record_op

        x = record_op("Flatten", [x], {"axis": 2})
        # transpose to [B, N, C] (ONNX transpose uses perm)
        x = transpose(x, perm=[0, 2, 1])
        return x


class Block:
    """Transformer Block."""

    def __init__(  # noqa: D107
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        prefix: str = "",
    ):
        self.prefix = prefix
        self.dim = dim
        self.norm1 = LayerNormalization((dim,), epsilon=1e-6)
        self.attn = MultiHeadAttention(num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = LayerNormalization((dim,), epsilon=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = Gemm(trans_b=1)
        self.act = Gelu()
        self.fc2 = Gemm(trans_b=1)
        self.mlp_hidden_dim = mlp_hidden_dim

    def __call__(self, x: Tensor) -> Tensor:  # noqa: D102
        identity = x
        x = self.norm1(
            x,
            get_param(f"{self.prefix}.norm1.weight", [self.dim]),
            get_param(f"{self.prefix}.norm1.bias", [self.dim]),
        )
        x = self.attn(x, x, x)
        x = add(x, identity)

        identity = x
        x = self.norm2(
            x,
            get_param(f"{self.prefix}.norm2.weight", [self.dim]),
            get_param(f"{self.prefix}.norm2.bias", [self.dim]),
        )
        x = self.fc1(
            x,
            get_param(f"{self.prefix}.fc1.weight", [self.mlp_hidden_dim, self.dim]),
            get_param(f"{self.prefix}.fc1.bias", [self.mlp_hidden_dim]),
        )
        x = self.act(x)
        x = self.fc2(
            x,
            get_param(f"{self.prefix}.fc2.weight", [self.dim, self.mlp_hidden_dim]),
            get_param(f"{self.prefix}.fc2.bias", [self.dim]),
        )
        x = add(x, identity)
        return x


class VisionTransformer:
    """Vision Transformer."""

    def __init__(  # noqa: D107
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ):
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            prefix="patch_embed",
        )
        self.blocks = [
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                prefix=f"blocks.{i}",
            )
            for i in range(depth)
        ]
        self.norm = LayerNormalization((embed_dim,), epsilon=1e-6)
        self.head = Gemm(trans_b=1)
        self.num_patches = self.patch_embed.num_patches

    def __call__(self, x: Tensor) -> Tensor:  # noqa: D102
        x = self.patch_embed(x)

        cls_token = get_param("cls_token", [1, 1, self.embed_dim])
        # Concat cls_token along axis 1 (seq_len)
        # Note: in real models cls_token is expanded to batch size, but for AST build we just concat
        x = concat([cls_token, x], axis=1)

        pos_embed = get_param("pos_embed", [1, self.num_patches + 1, self.embed_dim])
        x = add(x, pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(
            x, get_param("norm.weight", [self.embed_dim]), get_param("norm.bias", [self.embed_dim])
        )

        # Take the cls_token output (index 0)
        from onnx9000.core.ops import constant, record_op

        starts = constant([0], dtype=7)
        ends = constant([1], dtype=7)
        axes = constant([1], dtype=7)
        x = slice_op(x, starts, ends, axes)

        x = record_op("Flatten", [x], {"axis": 1})  # flatten spatial dims

        x = self.head(
            x,
            get_param("head.weight", [self.num_classes, self.embed_dim]),
            get_param("head.bias", [self.num_classes]),
        )
        return x


def vit_base_patch16_224(**kwargs: Any) -> VisionTransformer:  # noqa: D103
    return VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
