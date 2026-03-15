"""Test Pooling."""

from onnx9000 import GraphBuilder, Tracing
from onnx9000.frontends.frontend.nn.pool import (
    MaxPool1d,
    MaxPool2d,
    AvgPool1d,
    AvgPool2d,
)
from onnx9000.frontends.frontend.tensor import Tensor
from onnx9000.core.dtypes import DType


def test_pool():
    """Provides semantic functionality and verification."""
    mp1 = MaxPool1d(3, stride=2, padding=1)
    mp2 = MaxPool2d(3, ceil_mode=True)
    ap1 = AvgPool1d(3)
    ap2 = AvgPool2d(3, count_include_pad=False)
    assert mp1.kernel_size == (3,)
    assert mp2.stride == (3, 3)
    builder = GraphBuilder("dummy")
    with Tracing(builder):
        x1 = Tensor((20, 16, 50), DType.FLOAT32, "x1")
        x2 = Tensor((20, 16, 50, 50), DType.FLOAT32, "x2")
        y1 = mp1(x1)
        y2 = mp2(x2)
        y3 = ap1(x1)
        y4 = ap2(x2)
    assert y1 is not None
    assert y2 is not None
    assert y3 is not None
    assert y4 is not None
    mp3 = MaxPool1d((3,))
    mp4 = MaxPool2d((3, 3))
    ap3 = AvgPool1d((3,))
    ap4 = AvgPool2d((3, 3))


def test_adaptive_pool():
    """Provides semantic functionality and verification."""
    from onnx9000.frontends.frontend.nn.pool import AdaptiveAvgPool2d

    aap = AdaptiveAvgPool2d((1, 1))
    from onnx9000 import GraphBuilder, Tracing

    builder = GraphBuilder("dummy")
    with Tracing(builder):
        x = Tensor((20, 16, 50, 50), DType.FLOAT32, "x")
        y = aap(x)
    assert y is not None
