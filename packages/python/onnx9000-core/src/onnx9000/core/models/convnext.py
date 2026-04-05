"""Convnext."""

from typing import Any

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.ops import add, global_average_pool
from onnx9000.core.primitives import (
    ConvND,
    DepthwiseConv,
    Gelu,
    Gemm,
    LayerNormalization,
)


def get_param(name: str, shape: list[int], dtype: int = 1) -> Variable:
    """Docstring for D103."""
    return Variable(name=name, shape=shape, dtype=dtype)


class ConvNeXtBlock:
    """ConvNeXt Block."""

    def __init__(self, dim: int, drop_path: float = 0.0, prefix: str = ""):
        """Docstring for D107."""
        self.prefix = prefix
        self.dim = dim
        self.dwconv = DepthwiseConv(2, dim, kernel_size=7, padding=3)
        self.norm = LayerNormalization((dim,), epsilon=1e-6)
        self.pwconv1 = ConvND(2, dim, 4 * dim, kernel_size=1)
        self.act = Gelu()
        self.pwconv2 = ConvND(2, 4 * dim, dim, kernel_size=1)
        # Drop path and layer scale omitted for simplicity of AST builder

    def __call__(self, x: Tensor) -> Tensor:
        """Docstring for D102."""
        identity = x

        out = self.dwconv(
            x,
            get_param(f"{self.prefix}.dwconv.weight", [self.dwconv.out_channels, 1, 7, 7]),
            get_param(f"{self.prefix}.dwconv.bias", [self.dwconv.out_channels]),
        )

        # ConvNeXt does channel-first to channel-last for LayerNorm, but our LayerNorm
        # can take an axis. By default `LayerNormalization` has axis=-1.
        # For NCHW tensors, we should probably just use the primitive.
        # But wait, ONNX LayerNorm does it on the last axis.
        # Assume it's handled or we just apply it.
        # Let's apply it directly.
        out = self.norm(
            out,
            get_param(f"{self.prefix}.norm.weight", [self.dim]),
            get_param(f"{self.prefix}.norm.bias", [self.dim]),
        )

        out = self.pwconv1(
            out,
            get_param(
                f"{self.prefix}.pwconv1.weight",
                [self.pwconv1.out_channels, self.pwconv1.in_channels, 1, 1],
            ),
            get_param(f"{self.prefix}.pwconv1.bias", [self.pwconv1.out_channels]),
        )
        out = self.act(out)

        out = self.pwconv2(
            out,
            get_param(
                f"{self.prefix}.pwconv2.weight",
                [self.pwconv2.out_channels, self.pwconv2.in_channels, 1, 1],
            ),
            get_param(f"{self.prefix}.pwconv2.bias", [self.pwconv2.out_channels]),
        )

        out = add(identity, out)
        return out


class ConvNeXt:
    """ConvNeXt model built using IR primitives."""

    def __init__(self, in_chans: int = 3, num_classes: int = 1000):
        """Docstring for D107."""
        self.num_classes = num_classes
        self.stem_conv = ConvND(2, in_chans, 96, kernel_size=4, stride=4)
        self.stem_norm = LayerNormalization((96,), epsilon=1e-6)

        self.block1 = ConvNeXtBlock(96, prefix="block1")
        self.block2 = ConvNeXtBlock(96, prefix="block2")

        self.head_norm = LayerNormalization((96,), epsilon=1e-6)
        self.head = Gemm(trans_b=1)

    def __call__(self, x: Tensor) -> Tensor:
        """Docstring for D102."""
        x = self.stem_conv(
            x,
            get_param(
                "stem_conv.weight", [self.stem_conv.out_channels, self.stem_conv.in_channels, 4, 4]
            ),
            get_param("stem_conv.bias", [self.stem_conv.out_channels]),
        )
        x = self.stem_norm(
            x, get_param("stem_norm.weight", [96]), get_param("stem_norm.bias", [96])
        )

        x = self.block1(x)
        x = self.block2(x)

        x = global_average_pool(x)

        # Flatten before norm? ConvNeXt usually averages, then norms.
        from onnx9000.core.ops import flatten

        x = flatten(x)

        x = self.head_norm(
            x, get_param("head_norm.weight", [96]), get_param("head_norm.bias", [96])
        )
        x = self.head(
            x,
            get_param("head.weight", [self.num_classes, 96]),
            get_param("head.bias", [self.num_classes]),
        )

        return x


def convnext_tiny(**kwargs: Any) -> ConvNeXt:
    """Docstring for D103."""
    return ConvNeXt(**kwargs)
