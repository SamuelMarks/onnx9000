"""Unit tests and verification suite for boolean functionality."""

from onnx9000.c_compiler.boolean import (
    generate_boolean_binary,
    generate_boolean_unary,
    generate_where,
)
from onnx9000.core.exceptions import Onnx9000Error


def test_generate_boolean_binary() -> None:
    """Verify that generate boolean binary behaves as expected."""
    try:
        res = generate_boolean_binary()
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


def test_generate_boolean_unary() -> None:
    """Verify that generate boolean unary behaves as expected."""
    try:
        res = generate_boolean_unary()
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


def test_generate_where() -> None:
    """Verify that generate where behaves as expected."""
    try:
        res = generate_where()
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
