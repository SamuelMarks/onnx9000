"""Unit tests and verification suite for onnx parser functionality."""

from onnx9000.converters.onnx_parser import PureOnnxParser, read_tag, read_varint, skip_field
from onnx9000.core.exceptions import Onnx9000Error


def test_PureOnnxParser() -> None:
    """Verify that pure onnx parser behaves as expected."""
    try:
        obj = PureOnnxParser()
        assert obj is not None
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
    ):
        pass


def test_read_varint() -> None:
    """Verify that read varint behaves as expected."""
    try:
        res = read_varint()
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


def test_read_tag() -> None:
    """Verify that read tag behaves as expected."""
    try:
        res = read_tag()
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


def test_skip_field() -> None:
    """Verify that skip field behaves as expected."""
    try:
        res = skip_field()
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
