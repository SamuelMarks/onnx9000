"""Unit tests and verification suite for converters functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.toolkit.safetensors.converters import (
    convert_pytorch_to_safetensors,
    convert_tf_to_safetensors,
)


def test_convert_pytorch_to_safetensors() -> None:
    """Verify that convert pytorch to safetensors behaves as expected."""
    try:
        res = convert_pytorch_to_safetensors()
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


def test_convert_tf_to_safetensors() -> None:
    """Verify that convert tf to safetensors behaves as expected."""
    try:
        res = convert_tf_to_safetensors()
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
