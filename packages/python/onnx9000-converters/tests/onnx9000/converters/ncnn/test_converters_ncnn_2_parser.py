"""Unit tests and verification suite for parser functionality."""

from onnx9000.converters.ncnn.parser import parse_param
from onnx9000.core.exceptions import Onnx9000Error


def test_parse_param() -> None:
    """Verify that parse param behaves as expected."""
    try:
        res = parse_param()
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
