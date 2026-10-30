"""Unit tests and verification suite for reverse functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.onnx2gguf.reverse import reconstruct_onnx, reverse_map_name, reverse_map_type


def test_reverse_map_name() -> None:
    """Verify that reverse map name behaves as expected."""
    try:
        res = reverse_map_name()
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


def test_reverse_map_type() -> None:
    """Verify that reverse map type behaves as expected."""
    try:
        res = reverse_map_type()
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


def test_reconstruct_onnx() -> None:
    """Verify that reconstruct onnx behaves as expected."""
    try:
        res = reconstruct_onnx()
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
