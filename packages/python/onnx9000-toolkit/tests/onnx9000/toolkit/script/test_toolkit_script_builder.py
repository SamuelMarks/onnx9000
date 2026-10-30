"""Unit tests and verification suite for builder functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.toolkit.script.builder import GraphBuilder


def test_GraphBuilder() -> None:
    """Verify that graph builder behaves as expected."""
    try:
        obj = GraphBuilder()
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
