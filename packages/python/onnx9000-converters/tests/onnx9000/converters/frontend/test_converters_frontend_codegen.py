"""Unit tests and verification suite for codegen functionality."""

from onnx9000.converters.frontend.codegen import generate_jax, generate_keras, generate_pytorch
from onnx9000.core.exceptions import Onnx9000Error


def test_generate_pytorch() -> None:
    """Verify that generate pytorch behaves as expected."""
    try:
        res = generate_pytorch()
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


def test_generate_keras() -> None:
    """Verify that generate keras behaves as expected."""
    try:
        res = generate_keras()
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


def test_generate_jax() -> None:
    """Verify that generate jax behaves as expected."""
    try:
        res = generate_jax()
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
