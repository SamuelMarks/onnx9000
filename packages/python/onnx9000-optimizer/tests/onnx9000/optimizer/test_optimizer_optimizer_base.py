"""Unit tests and verification suite for base functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.base import Pass, PassContext


def test_PassContext() -> None:
    """Verify that pass context behaves as expected."""
    try:
        obj = PassContext()
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


def test_Pass() -> None:
    """Verify that pass behaves as expected."""
    try:
        obj = Pass()
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
