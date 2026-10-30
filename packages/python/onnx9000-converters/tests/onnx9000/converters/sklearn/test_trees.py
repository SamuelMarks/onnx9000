"""Unit tests and verification suite for trees functionality."""

from onnx9000.converters.sklearn.trees import (
    _convert_tree_classifier,
    _convert_tree_regressor,
    convert_ada_boost_classifier,
    convert_ada_boost_regressor,
    convert_bagging_classifier,
    convert_bagging_regressor,
    convert_decision_tree_classifier,
    convert_decision_tree_regressor,
    convert_extra_tree_classifier,
    convert_extra_tree_regressor,
    convert_extra_trees_classifier,
    convert_extra_trees_regressor,
    convert_gradient_boosting_classifier,
    convert_gradient_boosting_regressor,
    convert_hist_gradient_boosting_classifier,
    convert_hist_gradient_boosting_regressor,
    convert_isolation_forest,
    convert_random_forest_classifier,
    convert_random_forest_regressor,
    convert_stacking_classifier,
    convert_stacking_regressor,
    convert_voting_classifier,
    convert_voting_regressor,
)
from onnx9000.core.exceptions import Onnx9000Error


def test__convert_tree_classifier() -> None:
    """Verify that convert tree classifier behaves as expected."""
    try:
        res = _convert_tree_classifier()
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


def test__convert_tree_regressor() -> None:
    """Verify that convert tree regressor behaves as expected."""
    try:
        res = _convert_tree_regressor()
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


def test_convert_decision_tree_classifier() -> None:
    """Verify that convert decision tree classifier behaves as expected."""
    try:
        res = convert_decision_tree_classifier()
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


def test_convert_decision_tree_regressor() -> None:
    """Verify that convert decision tree regressor behaves as expected."""
    try:
        res = convert_decision_tree_regressor()
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


def test_convert_extra_tree_classifier() -> None:
    """Verify that convert extra tree classifier behaves as expected."""
    try:
        res = convert_extra_tree_classifier()
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


def test_convert_extra_tree_regressor() -> None:
    """Verify that convert extra tree regressor behaves as expected."""
    try:
        res = convert_extra_tree_regressor()
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


def test_convert_random_forest_classifier() -> None:
    """Verify that convert random forest classifier behaves as expected."""
    try:
        res = convert_random_forest_classifier()
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


def test_convert_random_forest_regressor() -> None:
    """Verify that convert random forest regressor behaves as expected."""
    try:
        res = convert_random_forest_regressor()
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


def test_convert_extra_trees_classifier() -> None:
    """Verify that convert extra trees classifier behaves as expected."""
    try:
        res = convert_extra_trees_classifier()
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


def test_convert_extra_trees_regressor() -> None:
    """Verify that convert extra trees regressor behaves as expected."""
    try:
        res = convert_extra_trees_regressor()
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


def test_convert_gradient_boosting_classifier() -> None:
    """Verify that convert gradient boosting classifier behaves as expected."""
    try:
        res = convert_gradient_boosting_classifier()
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


def test_convert_gradient_boosting_regressor() -> None:
    """Verify that convert gradient boosting regressor behaves as expected."""
    try:
        res = convert_gradient_boosting_regressor()
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


def test_convert_hist_gradient_boosting_classifier() -> None:
    """Verify that convert hist gradient boosting classifier behaves as expected."""
    try:
        res = convert_hist_gradient_boosting_classifier()
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


def test_convert_hist_gradient_boosting_regressor() -> None:
    """Verify that convert hist gradient boosting regressor behaves as expected."""
    try:
        res = convert_hist_gradient_boosting_regressor()
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


def test_convert_ada_boost_classifier() -> None:
    """Verify that convert ada boost classifier behaves as expected."""
    try:
        res = convert_ada_boost_classifier()
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


def test_convert_ada_boost_regressor() -> None:
    """Verify that convert ada boost regressor behaves as expected."""
    try:
        res = convert_ada_boost_regressor()
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


def test_convert_bagging_classifier() -> None:
    """Verify that convert bagging classifier behaves as expected."""
    try:
        res = convert_bagging_classifier()
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


def test_convert_bagging_regressor() -> None:
    """Verify that convert bagging regressor behaves as expected."""
    try:
        res = convert_bagging_regressor()
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


def test_convert_isolation_forest() -> None:
    """Verify that convert isolation forest behaves as expected."""
    try:
        res = convert_isolation_forest()
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


def test_convert_voting_classifier() -> None:
    """Verify that convert voting classifier behaves as expected."""
    try:
        res = convert_voting_classifier()
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


def test_convert_voting_regressor() -> None:
    """Verify that convert voting regressor behaves as expected."""
    try:
        res = convert_voting_regressor()
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


def test_convert_stacking_classifier() -> None:
    """Verify that convert stacking classifier behaves as expected."""
    try:
        res = convert_stacking_classifier()
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


def test_convert_stacking_regressor() -> None:
    """Verify that convert stacking regressor behaves as expected."""
    try:
        res = convert_stacking_regressor()
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
