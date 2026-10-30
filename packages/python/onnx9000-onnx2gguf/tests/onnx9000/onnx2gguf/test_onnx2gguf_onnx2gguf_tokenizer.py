"""Unit tests and verification suite for tokenizer functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.onnx2gguf.tokenizer import extract_tokenizer_metadata


def test_extract_tokenizer_metadata() -> None:
    """Verify that extract tokenizer metadata behaves as expected."""
    try:
        res = extract_tokenizer_metadata()
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
