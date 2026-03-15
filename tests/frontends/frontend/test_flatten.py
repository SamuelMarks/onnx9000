"""Test Flatten."""

from onnx9000 import GraphBuilder, Tracing
from onnx9000.frontends.frontend.nn.flatten import Flatten, Unflatten
from onnx9000.frontends.frontend.tensor import Tensor
from onnx9000.core.dtypes import DType


def test_flatten():
    """Provides semantic functionality and verification."""
    fl = Flatten()
    ufl = Unflatten(1, (5, 10))
    builder = GraphBuilder("dummy")
    with Tracing(builder):
        x = Tensor((20, 50), DType.FLOAT32, "x")
        y1 = fl(x)
        y2 = ufl(x)
    assert y1 is not None
    assert y2 is not None
