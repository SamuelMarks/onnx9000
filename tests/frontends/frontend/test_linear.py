"""Test Linear."""

from onnx9000 import GraphBuilder, Tracing
from onnx9000.frontends.frontend.nn.linear import Linear
from onnx9000.frontends.frontend.tensor import Tensor
from onnx9000.core.dtypes import DType


def test_linear():
    """Provides semantic functionality and verification."""
    layer = Linear(10, 5, bias=True)
    layer_no_bias = Linear(10, 5, bias=False)
    assert layer.weight.shape == (5, 10)
    assert layer.bias.shape == (5,)
    assert layer_no_bias.bias is None
    builder = GraphBuilder("dummy")
    with Tracing(builder):
        x = Tensor((2, 10), DType.FLOAT32, "x")
        y = layer(x)
        y2 = layer_no_bias(x)
    assert y is not None
    assert y2 is not None
