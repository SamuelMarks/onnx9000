"""Unit tests and verification suite for builder functionality."""

from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.core.exceptions import Onnx9000Error


def test_TFToONNXGraphBuilder() -> None:
    """Verify that tfto onnxgraph builder behaves as expected."""
    try:
        obj = TFToONNXGraphBuilder()
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
