"""Reference implementations of standard models."""

from onnx9000.converters.frontend.nn.containers import Sequential
from onnx9000.converters.frontend.nn.conv import Conv2d
from onnx9000.converters.frontend.nn.embedding import Embedding
from onnx9000.converters.frontend.nn.functional import relu
from onnx9000.converters.frontend.nn.linear import Linear
from onnx9000.converters.frontend.nn.module import Module
from onnx9000.converters.frontend.nn.normalization import BatchNorm2d, LayerNorm
from onnx9000.converters.frontend.nn.pool import AdaptiveAvgPool2d, MaxPool2d


class BasicBlock(Module):
    """Class BasicBlock implementation."""

    expansion = 1

    def __init__(
        self,
        inplanes: int | float | str | bool | bytes | list | dict | tuple | None,
        planes: int | float | str | bool | bytes | list | dict | tuple | None,
        stride: int | float | str | bool | bytes | list | dict | tuple | None = 1,
        downsample: int | float | str | bool | bytes | list | dict | tuple | None = None,
    ) -> None:
        """Initialize BasicBlock with configured parameters and state.

        Args:
            inplanes: Parameter specifying inplanes.
            planes: Parameter specifying planes.
            stride: Parameter specifying stride.
            downsample: Parameter specifying downsample.
        """
        super().__init__()
        self.conv1 = Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample

    def forward(
        self, x: int | float | str | bool | bytes | list | dict | tuple | None
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute forward on BasicBlock.

        Args:
            x: Parameter specifying x.

        Returns:
            Result of forward.
        """
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = relu(out)
        return out


class ResNet18(Module):
    """Class ResNet18 implementation."""

    def __init__(
        self, num_classes: int | float | str | bool | bytes | list | dict | tuple | None = 1000
    ) -> None:
        """Initialize ResNet18 with configured parameters and state.

        Args:
            num_classes: Parameter specifying num classes.
        """
        super().__init__()
        self.inplanes = 64
        self.conv1 = Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(self.inplanes)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512, num_classes)

    def _make_layer(
        self,
        planes: int | float | str | bool | bytes | list | dict | tuple | None,
        blocks: int | float | str | bool | bytes | list | dict | tuple | None,
        stride: int | float | str | bool | bytes | list | dict | tuple | None = 1,
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute make layer on ResNet18.

        Args:
            planes: Parameter specifying planes.
            blocks: Parameter specifying blocks.
            stride: Parameter specifying stride.

        Returns:
            Result of make layer.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = Sequential(
                Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return Sequential(*layers)

    def forward(
        self, x: int | float | str | bool | bytes | list | dict | tuple | None
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute forward on ResNet18.

        Args:
            x: Parameter specifying x.

        Returns:
            Result of forward.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class MobileNetV2(Module):
    """Class MobileNetV2 implementation."""

    def __init__(
        self, num_classes: int | float | str | bool | bytes | list | dict | tuple | None = 1000
    ) -> None:
        """Initialize MobileNetV2 with configured parameters and state.

        Args:
            num_classes: Parameter specifying num classes.
        """
        super().__init__()
        self.features = Sequential(
            Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(32),
            Conv2d(32, 1280, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(1280),
        )
        self.classifier = Sequential(Linear(1280, num_classes))

    def forward(
        self, x: int | float | str | bool | bytes | list | dict | tuple | None
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute forward on MobileNetV2.

        Args:
            x: Parameter specifying x.

        Returns:
            Result of forward.
        """
        x = self.features(x)
        x = x.mean((2, 3))
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class GPT2Block(Module):
    """Class GPT2Block implementation."""

    def __init__(
        self,
        d_model: int | float | str | bool | bytes | list | dict | tuple | None = 768,
        n_head: int | float | str | bool | bytes | list | dict | tuple | None = 12,
    ) -> None:
        """Initialize GPT2Block with configured parameters and state.

        Args:
            d_model: Parameter specifying d model.
            n_head: Parameter specifying n head.
        """
        super().__init__()
        self.ln_1 = LayerNorm(d_model)
        self.attn_c_attn = Linear(d_model, 3 * d_model)
        self.attn_c_proj = Linear(d_model, d_model)
        self.ln_2 = LayerNorm(d_model)
        self.mlp_c_fc = Linear(d_model, 4 * d_model)
        self.mlp_c_proj = Linear(4 * d_model, d_model)

    def forward(
        self, x: int | float | str | bool | bytes | list | dict | tuple | None
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute forward on GPT2Block.

        Args:
            x: Parameter specifying x.

        Returns:
            Result of forward.
        """
        a = self.ln_1(x)
        a = self.attn_c_attn(a)
        a = self.attn_c_proj(a)
        x = x + a
        m = self.ln_2(x)
        m = self.mlp_c_fc(m)
        m = relu(m)
        m = self.mlp_c_proj(m)
        x = x + m
        return x


class GPT2(Module):
    """Class GPT2 implementation."""

    def __init__(
        self,
        vocab_size: int | float | str | bool | bytes | list | dict | tuple | None = 50257,
        d_model: int | float | str | bool | bytes | list | dict | tuple | None = 768,
    ) -> None:
        """Initialize GPT2 with configured parameters and state.

        Args:
            vocab_size: Parameter specifying vocab size.
            d_model: Parameter specifying d model.
        """
        super().__init__()
        self.wte = Embedding(vocab_size, d_model)
        self.wpe = Embedding(1024, d_model)
        self.blocks = Sequential(*[GPT2Block(d_model) for _ in range(2)])
        self.ln_f = LayerNorm(d_model)

    def forward(
        self, idx: int | float | str | bool | bytes | list | dict | tuple | None
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute forward on GPT2.

        Args:
            idx: Parameter specifying idx.

        Returns:
            Result of forward.
        """
        x = self.wte(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        return x
