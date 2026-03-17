"""Test Identity."""

from onnx9000.core.dtypes import DType
from onnx9000.converters.frontend.builder import GraphBuilder, Tracing
from onnx9000.converters.frontend.nn.identity import Identity
from onnx9000.converters.frontend.tensor import Tensor


def test_identity() -> None:
    """Tests the test_identity functionality."""
    ident = Identity()
    builder = GraphBuilder("dummy")
    with Tracing(builder):
        x = Tensor((10,), DType.FLOAT32, "x")
        y = ident(x)
    assert y is not None
