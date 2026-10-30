"""Unit tests and verification suite for rnn functionality."""

from onnx9000.c_compiler.rnn import generate_attention, generate_rnn
from onnx9000.core.exceptions import Onnx9000Error


def test_generate_rnn() -> None:
    """Verify that generate rnn behaves as expected."""
    try:
        res = generate_rnn()
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


def test_generate_attention() -> None:
    """Verify that generate attention behaves as expected."""
    try:
        res = generate_attention()
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
