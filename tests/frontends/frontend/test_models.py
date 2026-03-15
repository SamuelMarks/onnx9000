"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.frontends.frontend.models import (
    BasicBlock,
    ResNet18,
    MobileNetV2,
    GPT2Block,
    GPT2,
)
from onnx9000.frontends.frontend.tensor import Tensor
from onnx9000.core.dtypes import DType
from onnx9000.frontends.frontend.builder import GraphBuilder, Tracing


def test_basic_block():
    """Provides semantic functionality and verification."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t = Tensor((1, 64, 56, 56), DType.FLOAT32)
        m = BasicBlock(64, 64)
        m(t)
        from onnx9000.frontends.frontend.nn.containers import Sequential
        from onnx9000.frontends.frontend.nn.conv import Conv2d

        down = Sequential(Conv2d(64, 128, kernel_size=1, stride=2, bias=False))
        m2 = BasicBlock(64, 128, stride=2, downsample=down)
        m2(t)


def test_resnet18():
    """Provides semantic functionality and verification."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t = Tensor((1, 3, 224, 224), DType.FLOAT32)
        m = ResNet18()
        m(t)


def test_mobilenetv2():
    """Provides semantic functionality and verification."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t = Tensor((1, 3, 224, 224), DType.FLOAT32)
        m = MobileNetV2()
        m(t)


def test_gpt2_block():
    """Provides semantic functionality and verification."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t = Tensor((1, 10, 768), DType.FLOAT32)
        m = GPT2Block(d_model=768)
        m(t)


def test_gpt2():
    """Provides semantic functionality and verification."""
    gb = GraphBuilder("mock")
    with Tracing(gb):
        t = Tensor((1, 10), DType.INT64)
        m = GPT2()
        m(t)
