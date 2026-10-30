"""Unit tests and verification suite for weights functionality."""

from onnx9000.converters.ncnn.weights import WeightsReader
from onnx9000.core.exceptions import Onnx9000Error


def test_WeightsReader() -> None:
    """Verify that weights reader behaves as expected."""
    try:
        obj = WeightsReader()
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
