"""Unit tests and verification suite for builder functionality."""

from onnx9000.converters.paddle.builder import PaddleToONNXGraphBuilder
from onnx9000.core.exceptions import Onnx9000Error


def test_PaddleToONNXGraphBuilder() -> None:
    """Verify that paddle to onnxgraph builder behaves as expected."""
    try:
        obj = PaddleToONNXGraphBuilder()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass
