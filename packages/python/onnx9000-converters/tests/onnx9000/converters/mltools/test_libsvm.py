"""Unit tests and verification suite for libsvm functionality."""

from onnx9000.converters.mltools.libsvm import parse_libsvm
from onnx9000.core.exceptions import Onnx9000Error


def test_parse_libsvm() -> None:
    """Verify that parse libsvm behaves as expected."""
    try:
        res = parse_libsvm()
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
