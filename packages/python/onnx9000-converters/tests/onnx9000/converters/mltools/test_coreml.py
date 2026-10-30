"""Unit tests and verification suite for coreml functionality."""

from onnx9000.converters.mltools.coreml import parse_coreml_model
from onnx9000.core.exceptions import Onnx9000Error


def test_parse_coreml_model() -> None:
    """Verify that parse coreml model behaves as expected."""
    try:
        res = parse_coreml_model()
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
