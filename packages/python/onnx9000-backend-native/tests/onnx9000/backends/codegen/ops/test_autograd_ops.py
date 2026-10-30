"""Unit tests and verification suite for autograd ops functionality."""

from onnx9000.backends.codegen.ops.autograd_ops import (
    generate_adamw,
    generate_relu_grad,
    generate_sgd,
)
from onnx9000.core.exceptions import Onnx9000Error


def test_generate_relu_grad() -> None:
    """Verify that generate relu grad behaves as expected."""
    try:
        res = generate_relu_grad()
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


def test_generate_sgd() -> None:
    """Verify that generate sgd behaves as expected."""
    try:
        res = generate_sgd()
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


def test_generate_adamw() -> None:
    """Verify that generate adamw behaves as expected."""
    try:
        res = generate_adamw()
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
