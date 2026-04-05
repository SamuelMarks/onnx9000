"""Efficientnet."""

from typing import Any

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.ops import add, flatten, global_average_pool, mul, reshape
from onnx9000.core.primitives import (
    BatchNormalization,
    ConvND,
    DepthwiseConv,
    Gemm,
    Sigmoid,
    Silu,
)


def get_param(name: str, shape: list[int], dtype: int = 1) -> Variable:  # noqa: D103
    """Get param."""
    return Variable(name=name, shape=shape, dtype=dtype)


class SqueezeExcitation:
    """Squeeze-and-Excitation block."""

    def __init__(self, in_channels: int, squeeze_channels: int, prefix: str = ""):  # noqa: D107
        """Init."""
        self.prefix = prefix
        self.in_channels = in_channels
        self.squeeze_channels = squeeze_channels
        self.fc1 = Gemm(trans_b=1)
        self.act = Silu()
        self.fc2 = Gemm(trans_b=1)
        self.scale_act = Sigmoid()

    def __call__(self, x: Tensor) -> Tensor:  # noqa: D102
        """Call."""
        scale = global_average_pool(x)
        scale = flatten(scale)

        fc1_w = get_param(f"{self.prefix}.fc1.weight", [self.squeeze_channels, self.in_channels])
        fc1_b = get_param(f"{self.prefix}.fc1.bias", [self.squeeze_channels])
        scale = self.fc1(scale, fc1_w, fc1_b)
        scale = self.act(scale)

        fc2_w = get_param(f"{self.prefix}.fc2.weight", [self.in_channels, self.squeeze_channels])
        fc2_b = get_param(f"{self.prefix}.fc2.bias", [self.in_channels])
        scale = self.fc2(scale, fc2_w, fc2_b)
        scale = self.scale_act(scale)

        # Reshape to [N, C, 1, 1] to broadcast with x
        # The input x shape is [N, C, H, W]
        # In ONNX, a reshape needs a shape tensor
        from onnx9000.core.ops import constant

        shape_tensor = constant([-1, self.in_channels, 1, 1], dtype=7)  # int64
        scale = reshape(scale, shape_tensor)

        return mul(x, scale)


class MBConv:
    """Mobile Inverted Bottleneck Convolution."""

    def __init__(  # noqa: D107
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        stride: int,
        kernel_size: int,
        prefix: str = "",
    ):
        """Init."""
        self.prefix = prefix
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        hidden_dim = in_channels * expand_ratio
        self.expand_ratio = expand_ratio

        if expand_ratio != 1:
            self.expand_conv = ConvND(2, in_channels, hidden_dim, kernel_size=1, bias=False)
            self.bn0 = BatchNormalization(hidden_dim)
            self.act0 = Silu()

        self.depthwise_conv = DepthwiseConv(
            2,
            hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn1 = BatchNormalization(hidden_dim)
        self.act1 = Silu()

        self.se = SqueezeExcitation(hidden_dim, max(1, in_channels // 4), prefix=f"{prefix}.se")

        self.project_conv = ConvND(2, hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn2 = BatchNormalization(out_channels)

    def __call__(self, x: Tensor) -> Tensor:  # noqa: D102
        """Call."""
        identity = x

        if self.expand_ratio != 1:
            x = self.expand_conv(
                x,
                get_param(
                    f"{self.prefix}.expand_conv.weight",
                    [self.expand_conv.out_channels, self.expand_conv.in_channels, 1, 1],
                ),
            )
            x = self.bn0(
                x,
                get_param(f"{self.prefix}.bn0.weight", [self.bn0.num_features]),
                get_param(f"{self.prefix}.bn0.bias", [self.bn0.num_features]),
                get_param(f"{self.prefix}.bn0.running_mean", [self.bn0.num_features]),
                get_param(f"{self.prefix}.bn0.running_var", [self.bn0.num_features]),
            )
            x = self.act0(x)

        x = self.depthwise_conv(
            x,
            get_param(
                f"{self.prefix}.depthwise_conv.weight",
                [
                    self.depthwise_conv.out_channels,
                    1,
                    self.depthwise_conv.kernel_size,
                    self.depthwise_conv.kernel_size,
                ],
            ),
        )
        x = self.bn1(
            x,
            get_param(f"{self.prefix}.bn1.weight", [self.bn1.num_features]),
            get_param(f"{self.prefix}.bn1.bias", [self.bn1.num_features]),
            get_param(f"{self.prefix}.bn1.running_mean", [self.bn1.num_features]),
            get_param(f"{self.prefix}.bn1.running_var", [self.bn1.num_features]),
        )
        x = self.act1(x)

        x = self.se(x)

        x = self.project_conv(
            x,
            get_param(
                f"{self.prefix}.project_conv.weight",
                [self.project_conv.out_channels, self.project_conv.in_channels, 1, 1],
            ),
        )
        x = self.bn2(
            x,
            get_param(f"{self.prefix}.bn2.weight", [self.bn2.num_features]),
            get_param(f"{self.prefix}.bn2.bias", [self.bn2.num_features]),
            get_param(f"{self.prefix}.bn2.running_mean", [self.bn2.num_features]),
            get_param(f"{self.prefix}.bn2.running_var", [self.bn2.num_features]),
        )

        if self.use_res_connect:
            x = add(identity, x)

        return x


class EfficientNet:
    """EfficientNet implementation built using IR macros/primitives."""

    def __init__(self, num_classes: int = 1000):  # noqa: D107
        """Init."""
        self.num_classes = num_classes
        self.stem_conv = ConvND(2, 3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.stem_bn = BatchNormalization(32)
        self.stem_act = Silu()

        self.block1 = MBConv(32, 16, expand_ratio=1, stride=1, kernel_size=3, prefix="block1")
        self.block2 = MBConv(16, 24, expand_ratio=6, stride=2, kernel_size=3, prefix="block2")

        self.head_conv = ConvND(2, 24, 1280, kernel_size=1, bias=False)
        self.head_bn = BatchNormalization(1280)
        self.head_act = Silu()

        self.classifier = Gemm(trans_b=1)

    def __call__(self, x: Tensor) -> Tensor:  # noqa: D102
        """Call."""
        x = self.stem_conv(
            x,
            get_param(
                "stem_conv.weight", [self.stem_conv.out_channels, self.stem_conv.in_channels, 3, 3]
            ),
        )
        x = self.stem_bn(
            x,
            get_param("stem_bn.weight", [self.stem_bn.num_features]),
            get_param("stem_bn.bias", [self.stem_bn.num_features]),
            get_param("stem_bn.running_mean", [self.stem_bn.num_features]),
            get_param("stem_bn.running_var", [self.stem_bn.num_features]),
        )
        x = self.stem_act(x)

        x = self.block1(x)
        x = self.block2(x)

        x = self.head_conv(
            x,
            get_param(
                "head_conv.weight", [self.head_conv.out_channels, self.head_conv.in_channels, 1, 1]
            ),
        )
        x = self.head_bn(
            x,
            get_param("head_bn.weight", [self.head_bn.num_features]),
            get_param("head_bn.bias", [self.head_bn.num_features]),
            get_param("head_bn.running_mean", [self.head_bn.num_features]),
            get_param("head_bn.running_var", [self.head_bn.num_features]),
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


def efficientnet_b0(**kwargs: Any) -> EfficientNet:  # noqa: D103
    """Efficientnet b0."""
    return EfficientNet(**kwargs)
