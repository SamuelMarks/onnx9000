"""Reference implementations of standard models."""

from onnx9000.frontend.frontend.nn.containers import Sequential
from onnx9000.frontend.frontend.nn.conv import Conv2d
from onnx9000.frontend.frontend.nn.embedding import Embedding
from onnx9000.frontend.frontend.nn.functional import relu
from onnx9000.frontend.frontend.nn.linear import Linear
from onnx9000.frontend.frontend.nn.module import Module
from onnx9000.frontend.frontend.nn.normalization import BatchNorm2d, LayerNorm
from onnx9000.frontend.frontend.nn.pool import AdaptiveAvgPool2d, MaxPool2d


class BasicBlock(Module):
    """Class BasicBlock implementation."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None) -> None:
        """Implements the __init__ method."""
        super().__init__()
        self.conv1 = Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        """Implements the forward method."""
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

    def __init__(self, num_classes=1000) -> None:
        """Implements the __init__ method."""
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

    def _make_layer(self, planes, blocks, stride=1):
        """Implements the _make_layer method."""
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

    def forward(self, x):
        """Implements the forward method."""
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

    def __init__(self, num_classes=1000) -> None:
        """Implements the __init__ method."""
        super().__init__()
        self.features = Sequential(
            Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(32),
            Conv2d(32, 1280, kernel_size=1, stride=1, bias=False),
            BatchNorm2d(1280),
        )
        self.classifier = Sequential(Linear(1280, num_classes))

    def forward(self, x):
        """Implements the forward method."""
        x = self.features(x)
        x = x.mean((2, 3))
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class GPT2Block(Module):
    """Class GPT2Block implementation."""

    def __init__(self, d_model=768, n_head=12) -> None:
        """Implements the __init__ method."""
        super().__init__()
        self.ln_1 = LayerNorm(d_model)
        self.attn_c_attn = Linear(d_model, 3 * d_model)
        self.attn_c_proj = Linear(d_model, d_model)
        self.ln_2 = LayerNorm(d_model)
        self.mlp_c_fc = Linear(d_model, 4 * d_model)
        self.mlp_c_proj = Linear(4 * d_model, d_model)

    def forward(self, x):
        """Implements the forward method."""
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

    def __init__(self, vocab_size=50257, d_model=768) -> None:
        """Implements the __init__ method."""
        super().__init__()
        self.wte = Embedding(vocab_size, d_model)
        self.wpe = Embedding(1024, d_model)
        self.blocks = Sequential(*[GPT2Block(d_model) for _ in range(2)])
        self.ln_f = LayerNorm(d_model)

    def forward(self, idx):
        """Implements the forward method."""
        x = self.wte(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        return x
