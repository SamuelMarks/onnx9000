"""Module docstring."""

import logging
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

from onnx9000.core.ir import Tensor, Variable
from onnx9000.core.ops import add, flatten, global_average_pool, max_pool
from onnx9000.core.primitives import BatchNormalization, ConvND, Gemm, Relu


def get_param(name: str, shape: list[int], dtype: int = 1) -> Variable:  # noqa: D103
    return Variable(name=name, shape=shape, dtype=dtype)


class BasicBlock:  # noqa: D101
    expansion: int = 1

    def __init__(  # noqa: D107
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: bool = False,
        prefix: str = "",
    ):
        self.prefix = prefix
        self.conv1 = ConvND(
            2, inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = BatchNormalization(planes)
        self.relu = Relu()
        self.conv2 = ConvND(2, planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNormalization(planes)
        self.downsample = downsample
        if self.downsample:
            self.downsample_conv = ConvND(
                2, inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False
            )
            self.downsample_bn = BatchNormalization(planes * self.expansion)

    def __call__(self, x: Tensor) -> Tensor:  # noqa: D102
        identity = x

        out = self.conv1(
            x,
            get_param(
                f"{self.prefix}.conv1.weight",
                [self.conv1.out_channels, self.conv1.in_channels, 3, 3],
            ),
        )
        out = self.bn1(
            out,
            get_param(f"{self.prefix}.bn1.weight", [self.bn1.num_features]),
            get_param(f"{self.prefix}.bn1.bias", [self.bn1.num_features]),
            get_param(f"{self.prefix}.bn1.running_mean", [self.bn1.num_features]),
            get_param(f"{self.prefix}.bn1.running_var", [self.bn1.num_features]),
        )
        out = self.relu(out)

        out = self.conv2(
            out,
            get_param(
                f"{self.prefix}.conv2.weight",
                [self.conv2.out_channels, self.conv2.in_channels, 3, 3],
            ),
        )
        out = self.bn2(
            out,
            get_param(f"{self.prefix}.bn2.weight", [self.bn2.num_features]),
            get_param(f"{self.prefix}.bn2.bias", [self.bn2.num_features]),
            get_param(f"{self.prefix}.bn2.running_mean", [self.bn2.num_features]),
            get_param(f"{self.prefix}.bn2.running_var", [self.bn2.num_features]),
        )

        if self.downsample:
            identity = self.downsample_conv(
                x,
                get_param(
                    f"{self.prefix}.downsample.0.weight",
                    [self.downsample_conv.out_channels, self.downsample_conv.in_channels, 1, 1],
                ),
            )
            identity = self.downsample_bn(
                identity,
                get_param(f"{self.prefix}.downsample.1.weight", [self.downsample_bn.num_features]),
                get_param(f"{self.prefix}.downsample.1.bias", [self.downsample_bn.num_features]),
                get_param(
                    f"{self.prefix}.downsample.1.running_mean", [self.downsample_bn.num_features]
                ),
                get_param(
                    f"{self.prefix}.downsample.1.running_var", [self.downsample_bn.num_features]
                ),
            )

        out = add(out, identity)
        out = self.relu(out)

        return out


class ResNet:
    """ResNet implementation built using IR macros/primitives."""

    def __init__(self, block_type: type, layers: list[int], num_classes: int = 1000):  # noqa: D107
        self.inplanes = 64
        self.conv1 = ConvND(2, 3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNormalization(self.inplanes)
        self.relu = Relu()

        self.layer1 = self._make_layer(block_type, 64, layers[0], prefix="layer1")
        self.layer2 = self._make_layer(block_type, 128, layers[1], stride=2, prefix="layer2")
        self.layer3 = self._make_layer(block_type, 256, layers[2], stride=2, prefix="layer3")
        self.layer4 = self._make_layer(block_type, 512, layers[3], stride=2, prefix="layer4")

        self.fc = Gemm(trans_b=1)  # PyTorch linear uses transB
        self.num_classes = num_classes

    def _make_layer(
        self, block_type: type, planes: int, blocks: int, stride: int = 1, prefix: str = ""
    ) -> list[Any]:
        downsample = False
        if stride != 1 or self.inplanes != planes * block_type.expansion:
            downsample = True

        layers = []
        layers.append(block_type(self.inplanes, planes, stride, downsample, prefix=f"{prefix}.0"))
        self.inplanes = planes * block_type.expansion
        for i in range(1, blocks):
            layers.append(block_type(self.inplanes, planes, prefix=f"{prefix}.{i}"))

        return layers

    def __call__(self, x: Tensor) -> Tensor:  # noqa: D102
        # Initial Convolution
        x = self.conv1(
            x, get_param("conv1.weight", [self.conv1.out_channels, self.conv1.in_channels, 7, 7])
        )
        x = self.bn1(
            x,
            get_param("bn1.weight", [self.bn1.num_features]),
            get_param("bn1.bias", [self.bn1.num_features]),
            get_param("bn1.running_mean", [self.bn1.num_features]),
            get_param("bn1.running_var", [self.bn1.num_features]),
        )
        x = self.relu(x)
        x = max_pool(x, kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])

        # Layers
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                x = block(x)

        # Classification Head
        x = global_average_pool(x)
        x = flatten(x)

        fc_w = get_param("fc.weight", [self.num_classes, 512 * BasicBlock.expansion])
        fc_b = get_param("fc.bias", [self.num_classes])

        x = self.fc(x, fc_w, fc_b)

        return x


def resnet18(**kwargs: Any) -> ResNet:  # noqa: D103
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet50(**kwargs: Any) -> ResNet:  # noqa: D103
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
