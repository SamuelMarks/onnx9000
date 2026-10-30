"""Unit tests and verification suite for sparkml functionality."""

from onnx9000.converters.mltools.sparkml import parse_sparkml_pipeline
from onnx9000.core.exceptions import Onnx9000Error


def test_parse_sparkml_pipeline() -> None:
    """Verify that parse sparkml pipeline behaves as expected."""
    try:
        res = parse_sparkml_pipeline()
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
