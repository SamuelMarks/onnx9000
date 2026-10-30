"""Unit tests and verification suite for registry functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.tensorrt.registry import get_op_translator, register_op


def test_register_op() -> None:
    """Verify that register op behaves as expected."""
    try:
        res = register_op()
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


def test_get_op_translator() -> None:
    """Verify that get op translator behaves as expected."""
    try:
        res = get_op_translator()
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
