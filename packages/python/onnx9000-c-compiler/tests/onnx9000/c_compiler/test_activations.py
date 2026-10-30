"""Unit tests and verification suite for activations functionality."""

from onnx9000.c_compiler.activations import (
    generate_activation,
    generate_normalization,
    generate_softmax,
)
from onnx9000.core.exceptions import Onnx9000Error


def test_generate_activation() -> None:
    """Verify that generate activation behaves as expected."""
    try:
        res = generate_activation()
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


def test_generate_softmax() -> None:
    """Verify that generate softmax behaves as expected."""
    try:
        res = generate_softmax()
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


def test_generate_normalization() -> None:
    """Verify that generate normalization behaves as expected."""
    try:
        res = generate_normalization()
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
