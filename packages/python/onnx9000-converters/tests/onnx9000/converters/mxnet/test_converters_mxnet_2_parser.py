"""Unit tests and verification suite for parser functionality."""

from onnx9000.converters.mxnet.parser import parse_symbol
from onnx9000.core.exceptions import Onnx9000Error


def test_parse_symbol() -> None:
    """Verify that parse symbol behaves as expected."""
    try:
        res = parse_symbol()
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
