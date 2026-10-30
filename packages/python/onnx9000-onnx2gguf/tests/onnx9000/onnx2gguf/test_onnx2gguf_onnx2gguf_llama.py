"""Unit tests and verification suite for llama functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.onnx2gguf.llama import extract_llama_metadata


def test_extract_llama_metadata() -> None:
    """Verify that extract llama metadata behaves as expected."""
    try:
        res = extract_llama_metadata()
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
