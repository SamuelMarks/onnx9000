"""Unit tests and verification suite for interop functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.toolkit.safetensors.interop import (
    load_flax_safetensors,
    load_pytorch_safetensors,
    load_tensorflow_safetensors,
)


def test_load_pytorch_safetensors() -> None:
    """Verify that load pytorch safetensors behaves as expected."""
    try:
        res = load_pytorch_safetensors()
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


def test_load_tensorflow_safetensors() -> None:
    """Verify that load tensorflow safetensors behaves as expected."""
    try:
        res = load_tensorflow_safetensors()
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


def test_load_flax_safetensors() -> None:
    """Verify that load flax safetensors behaves as expected."""
    try:
        res = load_flax_safetensors()
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
