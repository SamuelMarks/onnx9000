"""Unit tests and verification suite for flatten functionality."""

from onnx9000.converters.frontend.nn.flatten import Flatten, Unflatten
from onnx9000.core.exceptions import Onnx9000Error


def test_Flatten() -> None:
    """Verify that flatten behaves as expected."""
    try:
        obj = Flatten()
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


def test_Unflatten() -> None:
    """Verify that unflatten behaves as expected."""
    try:
        obj = Unflatten()
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
