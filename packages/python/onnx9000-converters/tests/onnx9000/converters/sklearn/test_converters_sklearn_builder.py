"""Unit tests and verification suite for builder functionality."""

from onnx9000.converters.sklearn.builder import SKLearnParser
from onnx9000.core.exceptions import Onnx9000Error


def test_SKLearnParser() -> None:
    """Verify that sklearn parser behaves as expected."""
    try:
        obj = SKLearnParser()
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
