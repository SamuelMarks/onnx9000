"""Module providing core logic and structural definitions."""

from onnx9000.core.dtypes import DType
from onnx9000.frontend.frontend.builder import GraphBuilder, Tracing
from onnx9000.frontend.frontend.models import GPT2, BasicBlock, GPT2Block, MobileNetV2, ResNet18
from onnx9000.frontend.frontend.tensor import Tensor


def test_basic_block() -> None:
    """Tests the test_basic_block functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t = Tensor((1, 64, 56, 56), DType.FLOAT32)
        m = BasicBlock(64, 64)
        m(t)
        from onnx9000.frontend.frontend.nn.containers import Sequential
        from onnx9000.frontend.frontend.nn.conv import Conv2d

        down = Sequential(Conv2d(64, 128, kernel_size=1, stride=2, bias=False))
        m2 = BasicBlock(64, 128, stride=2, downsample=down)
        m2(t)


def test_resnet18() -> None:
    """Tests the test_resnet18 functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t = Tensor((1, 3, 224, 224), DType.FLOAT32)
        m = ResNet18()
        m(t)


def test_mobilenetv2() -> None:
    """Tests the test_mobilenetv2 functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t = Tensor((1, 3, 224, 224), DType.FLOAT32)
        m = MobileNetV2()
        m(t)


def test_gpt2_block() -> None:
    """Tests the test_gpt2_block functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t = Tensor((1, 10, 768), DType.FLOAT32)
        m = GPT2Block(d_model=768)
        m(t)


def test_gpt2() -> None:
    """Tests the test_gpt2 functionality."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t = Tensor((1, 10), DType.INT64)
        m = GPT2()
        m(t)
