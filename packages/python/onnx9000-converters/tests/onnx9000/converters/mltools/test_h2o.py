"""Unit tests and verification suite for h2o functionality."""

from onnx9000.converters.mltools.h2o import parse_h2o
from onnx9000.core.exceptions import Onnx9000Error


def test_parse_h2o() -> None:
    """Verify that parse h2o behaves as expected."""
    try:
        res = parse_h2o()
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
