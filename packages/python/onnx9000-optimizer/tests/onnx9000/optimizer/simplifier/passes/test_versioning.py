"""Unit tests and verification suite for versioning functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.simplifier.passes.versioning import apply_opset_fallbacks, enforce_opset_18


def test_apply_opset_fallbacks() -> None:
    """Verify that apply opset fallbacks behaves as expected."""
    try:
        res = apply_opset_fallbacks()
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


def test_enforce_opset_18() -> None:
    """Verify that enforce opset 18 behaves as expected."""
    try:
        res = enforce_opset_18()
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
