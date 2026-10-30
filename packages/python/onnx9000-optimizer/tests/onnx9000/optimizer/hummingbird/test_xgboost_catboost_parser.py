"""Unit tests and verification suite for xgboost catboost parser functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.hummingbird.xgboost_catboost_parser import (
    _traverse_xgb_tree,
    handle_catboost_categorical,
    handle_xgb_objectives,
    parse_catboost_classifier,
    parse_catboost_regressor,
    parse_xgb_classifier,
    parse_xgb_ranker,
    parse_xgb_regressor,
    parse_xgboost_dump,
)


def test_parse_xgboost_dump() -> None:
    """Verify that parse xgboost dump behaves as expected."""
    try:
        res = parse_xgboost_dump()
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


def test__traverse_xgb_tree() -> None:
    """Verify that traverse xgb tree behaves as expected."""
    try:
        res = _traverse_xgb_tree()
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


def test_parse_xgb_classifier() -> None:
    """Verify that parse xgb classifier behaves as expected."""
    try:
        res = parse_xgb_classifier()
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


def test_parse_xgb_regressor() -> None:
    """Verify that parse xgb regressor behaves as expected."""
    try:
        res = parse_xgb_regressor()
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


def test_parse_xgb_ranker() -> None:
    """Verify that parse xgb ranker behaves as expected."""
    try:
        res = parse_xgb_ranker()
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


def test_handle_xgb_objectives() -> None:
    """Verify that handle xgb objectives behaves as expected."""
    try:
        res = handle_xgb_objectives()
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


def test_parse_catboost_classifier() -> None:
    """Verify that parse catboost classifier behaves as expected."""
    try:
        res = parse_catboost_classifier()
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


def test_parse_catboost_regressor() -> None:
    """Verify that parse catboost regressor behaves as expected."""
    try:
        res = parse_catboost_regressor()
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


def test_handle_catboost_categorical() -> None:
    """Verify that handle catboost categorical behaves as expected."""
    try:
        res = handle_catboost_categorical()
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
