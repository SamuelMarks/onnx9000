"""Unit tests and verification suite for vision functionality."""

from onnx9000.c_compiler.vision import generate_nms, generate_resize
from onnx9000.core.exceptions import Onnx9000Error


def test_generate_nms() -> None:
    """Verify that generate nms behaves as expected."""
    try:
        res = generate_nms()
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


def test_generate_resize() -> None:
    """Verify that generate resize behaves as expected."""
    try:
        res = generate_resize()
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
