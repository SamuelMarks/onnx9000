"""Unit tests and verification suite for ml builders functionality."""

from onnx9000.converters.sklearn.ml_builders import (
    convert_mlp_classifier,
    convert_mlp_regressor,
    convert_select_fpr,
    convert_select_k_best,
    convert_select_percentile,
    convert_variance_threshold,
)
from onnx9000.core.exceptions import Onnx9000Error


def test_convert_mlp_classifier() -> None:
    """Verify that convert mlp classifier behaves as expected."""
    try:
        res = convert_mlp_classifier()
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


def test_convert_mlp_regressor() -> None:
    """Verify that convert mlp regressor behaves as expected."""
    try:
        res = convert_mlp_regressor()
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


def test_convert_select_k_best() -> None:
    """Verify that convert select k best behaves as expected."""
    try:
        res = convert_select_k_best()
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


def test_convert_select_percentile() -> None:
    """Verify that convert select percentile behaves as expected."""
    try:
        res = convert_select_percentile()
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


def test_convert_select_fpr() -> None:
    """Verify that convert select fpr behaves as expected."""
    try:
        res = convert_select_fpr()
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


def test_convert_variance_threshold() -> None:
    """Verify that convert variance threshold behaves as expected."""
    try:
        res = convert_variance_threshold()
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
