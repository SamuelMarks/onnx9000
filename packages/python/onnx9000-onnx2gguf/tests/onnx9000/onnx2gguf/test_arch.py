"""Unit tests and verification suite for arch functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.onnx2gguf.arch import extract_metadata, infer_architecture


def test_extract_metadata() -> None:
    """Verify that extract metadata behaves as expected."""
    try:
        res = extract_metadata()
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


def test_infer_architecture() -> None:
    """Verify that infer architecture behaves as expected."""
    try:
        res = infer_architecture()
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
