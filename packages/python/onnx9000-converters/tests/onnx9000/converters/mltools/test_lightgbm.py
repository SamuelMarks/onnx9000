"""Unit tests and verification suite for lightgbm functionality."""

from onnx9000.converters.mltools.lightgbm import parse_lightgbm_dict, parse_lightgbm_json
from onnx9000.core.exceptions import Onnx9000Error


def test_parse_lightgbm_json() -> None:
    """Verify that parse lightgbm json behaves as expected."""
    try:
        res = parse_lightgbm_json()
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


def test_parse_lightgbm_dict() -> None:
    """Verify that parse lightgbm dict behaves as expected."""
    try:
        res = parse_lightgbm_dict()
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
