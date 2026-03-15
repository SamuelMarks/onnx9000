"""Test Identity."""

from onnx9000 import GraphBuilder, Tracing
from onnx9000.frontends.frontend.nn.identity import Identity
from onnx9000.frontends.frontend.tensor import Tensor
from onnx9000.core.dtypes import DType


def test_identity():
    """Provides semantic functionality and verification."""
    ident = Identity()
    builder = GraphBuilder("dummy")
    with Tracing(builder):
        x = Tensor((10,), DType.FLOAT32, "x")
        y = ident(x)
    assert y is not None
