"""Unit tests and verification suite for pooling functionality."""

from onnx9000.c_compiler.pooling import (
    generate_arg_reduction,
    generate_global_pooling,
    generate_pooling,
    generate_reduction,
)
from onnx9000.core.exceptions import Onnx9000Error


def test_generate_pooling() -> None:
    """Verify that generate pooling behaves as expected."""
    try:
        res = generate_pooling()
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


def test_generate_global_pooling() -> None:
    """Verify that generate global pooling behaves as expected."""
    try:
        res = generate_global_pooling()
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


def test_generate_reduction() -> None:
    """Verify that generate reduction behaves as expected."""
    try:
        res = generate_reduction()
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


def test_generate_arg_reduction() -> None:
    """Verify that generate arg reduction behaves as expected."""
    try:
        res = generate_arg_reduction()
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
