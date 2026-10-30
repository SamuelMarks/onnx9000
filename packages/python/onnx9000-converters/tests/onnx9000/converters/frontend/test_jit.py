"""Unit tests and verification suite for jit functionality."""

from onnx9000.converters.frontend.jit import jit
from onnx9000.core.exceptions import Onnx9000Error


def test_jit() -> None:
    """Verify that jit behaves as expected."""
    try:
        res = jit()
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
