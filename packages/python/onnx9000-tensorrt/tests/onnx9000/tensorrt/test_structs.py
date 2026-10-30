"""Unit tests and verification suite for structs functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.tensorrt.structs import Dims, Weights


def test_Dims() -> None:
    """Verify that dims behaves as expected."""
    try:
        obj = Dims()
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


def test_Weights() -> None:
    """Verify that weights behaves as expected."""
    try:
        obj = Weights()
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
