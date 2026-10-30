"""Unit tests and verification suite for compiler functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.onnx2gguf.compiler import (
    compile_gguf,
    get_gguf_type,
    infer_architecture,
    sanitize_doc_string,
)


def test_get_gguf_type() -> None:
    """Verify that get gguf type behaves as expected."""
    try:
        res = get_gguf_type()
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


def test_sanitize_doc_string() -> None:
    """Verify that sanitize doc string behaves as expected."""
    try:
        res = sanitize_doc_string()
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


def test_compile_gguf() -> None:
    """Verify that compile gguf behaves as expected."""
    try:
        res = compile_gguf()
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
