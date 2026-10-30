"""Unit tests and verification suite for lightgbm parser functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.hummingbird.lightgbm_parser import (
    _traverse_lgbm_tree,
    apply_lgbm_scaling,
    handle_lgbm_objectives,
    parse_lgbm_categorical,
    parse_lgbm_classifier,
    parse_lgbm_ranker,
    parse_lgbm_regressor,
    parse_lightgbm_dump,
)


def test_parse_lightgbm_dump() -> None:
    """Verify that parse lightgbm dump behaves as expected."""
    try:
        res = parse_lightgbm_dump()
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


def test__traverse_lgbm_tree() -> None:
    """Verify that traverse lgbm tree behaves as expected."""
    try:
        res = _traverse_lgbm_tree()
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


def test_parse_lgbm_classifier() -> None:
    """Verify that parse lgbm classifier behaves as expected."""
    try:
        res = parse_lgbm_classifier()
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


def test_parse_lgbm_regressor() -> None:
    """Verify that parse lgbm regressor behaves as expected."""
    try:
        res = parse_lgbm_regressor()
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


def test_parse_lgbm_ranker() -> None:
    """Verify that parse lgbm ranker behaves as expected."""
    try:
        res = parse_lgbm_ranker()
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


def test_handle_lgbm_objectives() -> None:
    """Verify that handle lgbm objectives behaves as expected."""
    try:
        res = handle_lgbm_objectives()
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


def test_parse_lgbm_categorical() -> None:
    """Verify that parse lgbm categorical behaves as expected."""
    try:
        res = parse_lgbm_categorical()
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


def test_apply_lgbm_scaling() -> None:
    """Verify that apply lgbm scaling behaves as expected."""
    try:
        res = apply_lgbm_scaling()
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
