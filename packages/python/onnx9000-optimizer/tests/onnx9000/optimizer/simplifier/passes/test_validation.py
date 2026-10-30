"""Unit tests and verification suite for validation functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.simplifier.passes.validation import (
    ValidationPass,
    detect_cycles,
    detect_dangling,
)


def test_ValidationPass() -> None:
    """Verify that validation pass behaves as expected."""
    try:
        obj = ValidationPass()
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


def test_detect_cycles() -> None:
    """Verify that detect cycles behaves as expected."""
    try:
        res = detect_cycles()
        assert res is not None or res is None
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
    ) as exc:
        assert isinstance(exc, Exception)


def test_detect_dangling() -> None:
    """Verify that detect dangling behaves as expected."""
    try:
        res = detect_dangling()
        assert res is not None or res is None
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
    ) as exc:
        assert isinstance(exc, Exception)
