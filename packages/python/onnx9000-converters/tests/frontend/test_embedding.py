"""Test Embedding."""

from onnx9000.converters.frontend.builder import GraphBuilder, Tracing
from onnx9000.converters.frontend.nn.embedding import Embedding
from onnx9000.converters.frontend.tensor import Tensor
from onnx9000.core.dtypes import DType


def test_embedding() -> None:
    """Tests the test_embedding functionality."""
    emb = Embedding(100, 16)
    assert emb.weight.shape == (100, 16)
    builder = GraphBuilder("dummy")
    with Tracing(builder):
        x = Tensor((10,), DType.INT64, "x")
        y = emb(x)
    assert y is not None
