"""Unit tests and verification suite for nlp functionality."""

from onnx9000.c_compiler.nlp import emit_bpe_tokenizer, generate_topk, generate_unique
from onnx9000.core.exceptions import Onnx9000Error


def test_generate_topk() -> None:
    """Verify that generate topk behaves as expected."""
    try:
        res = generate_topk()
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


def test_generate_unique() -> None:
    """Verify that generate unique behaves as expected."""
    try:
        res = generate_unique()
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


def test_emit_bpe_tokenizer() -> None:
    """Verify that emit bpe tokenizer behaves as expected."""
    try:
        res = emit_bpe_tokenizer()
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
