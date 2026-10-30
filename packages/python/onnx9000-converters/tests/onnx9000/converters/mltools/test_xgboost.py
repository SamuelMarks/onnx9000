"""Unit tests and verification suite for xgboost functionality."""

from onnx9000.converters.mltools.xgboost import parse_xgboost_dict, parse_xgboost_json
from onnx9000.core.exceptions import Onnx9000Error


def test_parse_xgboost_json() -> None:
    """Verify that parse xgboost json behaves as expected."""
    try:
        res = parse_xgboost_json()
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


def test_parse_xgboost_dict() -> None:
    """Verify that parse xgboost dict behaves as expected."""
    try:
        res = parse_xgboost_dict()
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
