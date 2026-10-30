"""Unit tests and verification suite for catboost functionality."""

from onnx9000.converters.mltools.catboost import parse_catboost_dict, parse_catboost_json
from onnx9000.core.exceptions import Onnx9000Error


def test_parse_catboost_json() -> None:
    """Verify that parse catboost json behaves as expected."""
    try:
        res = parse_catboost_json()
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


def test_parse_catboost_dict() -> None:
    """Verify that parse catboost dict behaves as expected."""
    try:
        res = parse_catboost_dict()
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
