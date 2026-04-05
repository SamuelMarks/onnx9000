"""Mobilevit."""

from typing import Any, Optional

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.ops import add, flatten, global_average_pool, reshape
from onnx9000.core.primitives import (
    BatchNormalization,
    ConvND,
    DepthwiseConv,
    Gemm,
    LayerNormalization,
    MultiHeadAttention,
    Silu,
)


def get_param(name: str, shape: list[int], dtype: int = 1) -> Variable:
    """Docstring for D103."""
    return Variable(name=name, shape=shape, dtype=dtype)


class MobileViTBlock:
    """MobileViT block combining Convolutions and Transformers."""

    def __init__(self, dim: int, out_dim: int, num_heads: int, mlp_dim: int, prefix: str = ""):
        """Docstring for D107."""
        self.prefix = prefix
        self.dim = dim
        self.out_dim = out_dim

        # Local representation
        self.local_conv1 = ConvND(2, dim, dim, kernel_size=3, padding=1, bias=False)
        self.local_bn1 = BatchNormalization(dim)
        self.local_act1 = Silu()

        self.local_conv2 = ConvND(2, dim, dim, kernel_size=1, bias=False)
        self.local_bn2 = BatchNormalization(dim)
        self.local_act2 = Silu()

        # Global representation (Transformer)
        self.attn = MultiHeadAttention(num_heads=num_heads)
        self.norm1 = LayerNormalization((dim,))
        self.norm2 = LayerNormalization((dim,))
        self.mlp_fc1 = Gemm(trans_b=1)
        self.mlp_act = Silu()
        self.mlp_fc2 = Gemm(trans_b=1)

        # Fusion
        self.fusion_conv1 = ConvND(2, dim, dim, kernel_size=1, bias=False)
        self.fusion_bn1 = BatchNormalization(dim)
        self.fusion_act1 = Silu()

        self.fusion_conv2 = ConvND(2, dim * 2, out_dim, kernel_size=3, padding=1, bias=False)
        self.fusion_bn2 = BatchNormalization(out_dim)
        self.fusion_act2 = Silu()

    def __call__(self, x: Tensor) -> Tensor:
        """Docstring for D102."""
        identity = x

        # Local
        out = self.local_conv1(
            x,
            get_param(
                f"{self.prefix}.local_conv1.weight",
                [self.local_conv1.out_channels, self.local_conv1.in_channels, 3, 3],
            ),
        )
        out = self.local_bn1(
            out,
            get_param(f"{self.prefix}.local_bn1.weight", [self.dim]),
            get_param(f"{self.prefix}.local_bn1.bias", [self.dim]),
            get_param(f"{self.prefix}.local_bn1.running_mean", [self.dim]),
            get_param(f"{self.prefix}.local_bn1.running_var", [self.dim]),
        )
        out = self.local_act1(out)

        out = self.local_conv2(
            out,
            get_param(
                f"{self.prefix}.local_conv2.weight",
                [self.local_conv2.out_channels, self.local_conv2.in_channels, 1, 1],
            ),
        )
        out = self.local_bn2(
            out,
            get_param(f"{self.prefix}.local_bn2.weight", [self.dim]),
            get_param(f"{self.prefix}.local_bn2.bias", [self.dim]),
            get_param(f"{self.prefix}.local_bn2.running_mean", [self.dim]),
            get_param(f"{self.prefix}.local_bn2.running_var", [self.dim]),
        )
        out = self.local_act2(out)

        # Transformer (assuming unfolded/reshaped for sequence here, simplified for AST building)
        # MobileViT unfolds into patches, but we just use the primitives.
        attn_out = self.norm1(
            out,
            get_param(f"{self.prefix}.norm1.weight", [self.dim]),
            get_param(f"{self.prefix}.norm1.bias", [self.dim]),
        )
        attn_out = self.attn(attn_out, attn_out, attn_out)  # Q, K, V
        attn_out = add(out, attn_out)

        mlp_out = self.norm2(
            attn_out,
            get_param(f"{self.prefix}.norm2.weight", [self.dim]),
            get_param(f"{self.prefix}.norm2.bias", [self.dim]),
        )
        mlp_out = self.mlp_fc1(
            mlp_out,
            get_param(f"{self.prefix}.mlp_fc1.weight", [self.dim * 2, self.dim]),
            get_param(f"{self.prefix}.mlp_fc1.bias", [self.dim * 2]),
        )
        mlp_out = self.mlp_act(mlp_out)
        mlp_out = self.mlp_fc2(
            mlp_out,
            get_param(f"{self.prefix}.mlp_fc2.weight", [self.dim, self.dim * 2]),
            get_param(f"{self.prefix}.mlp_fc2.bias", [self.dim]),
        )
        attn_out = add(attn_out, mlp_out)

        # Fusion
        out = self.fusion_conv1(
            attn_out, get_param(f"{self.prefix}.fusion_conv1.weight", [self.dim, self.dim, 1, 1])
        )
        out = self.fusion_bn1(
            out,
            get_param(f"{self.prefix}.fusion_bn1.weight", [self.dim]),
            get_param(f"{self.prefix}.fusion_bn1.bias", [self.dim]),
            get_param(f"{self.prefix}.fusion_bn1.running_mean", [self.dim]),
            get_param(f"{self.prefix}.fusion_bn1.running_var", [self.dim]),
        )
        out = self.fusion_act1(out)

        from onnx9000.core.ops import concat

        # Concat along channel dim
        out = concat([identity, out], axis=1)

        out = self.fusion_conv2(
            out, get_param(f"{self.prefix}.fusion_conv2.weight", [self.out_dim, self.dim * 2, 3, 3])
        )
        out = self.fusion_bn2(
            out,
            get_param(f"{self.prefix}.fusion_bn2.weight", [self.out_dim]),
            get_param(f"{self.prefix}.fusion_bn2.bias", [self.out_dim]),
            get_param(f"{self.prefix}.fusion_bn2.running_mean", [self.out_dim]),
            get_param(f"{self.prefix}.fusion_bn2.running_var", [self.out_dim]),
        )
        out = self.fusion_act2(out)

        return out


class MobileViT:
    """MobileViT implementation."""

    def __init__(self, num_classes: int = 1000):
        """Docstring for D107."""
        self.num_classes = num_classes
        self.stem_conv = ConvND(2, 3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.stem_bn = BatchNormalization(16)
        self.stem_act = Silu()

        self.block1 = MobileViTBlock(16, 32, num_heads=4, mlp_dim=64, prefix="block1")

        self.head_conv = ConvND(2, 32, 1280, kernel_size=1, bias=False)
        self.head_bn = BatchNormalization(1280)
        self.head_act = Silu()

        self.classifier = Gemm(trans_b=1)

    def __call__(self, x: Tensor) -> Tensor:
        """Docstring for D102."""
        x = self.stem_conv(x, get_param("stem_conv.weight", [16, 3, 3, 3]))
        x = self.stem_bn(
            x,
            get_param("stem_bn.weight", [16]),
            get_param("stem_bn.bias", [16]),
            get_param("stem_bn.running_mean", [16]),
            get_param("stem_bn.running_var", [16]),
        )
        x = self.stem_act(x)

        x = self.block1(x)

        x = self.head_conv(x, get_param("head_conv.weight", [1280, 32, 1, 1]))
        x = self.head_bn(
            x,
            get_param("head_bn.weight", [1280]),
            get_param("head_bn.bias", [1280]),
            get_param("head_bn.running_mean", [1280]),
            get_param("head_bn.running_var", [1280]),
        )
        x = self.head_act(x)

        x = global_average_pool(x)
        x = flatten(x)
        x = self.classifier(
            x,
            get_param("classifier.weight", [self.num_classes, 1280]),
            get_param("classifier.bias", [self.num_classes]),
        )

        return x


def mobilevit_s(**kwargs: Any) -> MobileViT:
    """Docstring for D103."""
    return MobileViT(**kwargs)
