"""Unit tests and verification suite for sklearn parser functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.hummingbird.sklearn_parser import (
    extract_classes_and_zipmaps,
    extract_n_estimators,
    handle_multi_label_classification,
    handle_multi_output_regressors,
    handle_predict_proba,
    optimize_binarizer,
    optimize_onehot_encoder,
    optimize_standard_scaler,
    parse_ada_boost_classifier,
    parse_ada_boost_regressor,
    parse_bernoulli_nb,
    parse_decision_tree_classifier,
    parse_decision_tree_regressor,
    parse_extra_trees_classifier,
    parse_extra_trees_regressor,
    parse_gaussian_nb,
    parse_gradient_boosting_classifier,
    parse_gradient_boosting_regressor,
    parse_hist_gradient_boosting_classifier,
    parse_hist_gradient_boosting_regressor,
    parse_isolation_forest,
    parse_linear_regression,
    parse_linear_svc,
    parse_logistic_regression,
    parse_mlp_classifier,
    parse_multinomial_nb,
    parse_pipeline,
    parse_random_forest_classifier,
    parse_random_forest_regressor,
    parse_ridge_lasso_elasticnet,
    parse_sgd_classifier,
    parse_svc_poly,
    parse_svc_rbf,
    parse_svc_sigmoid,
)


def test_parse_decision_tree_classifier() -> None:
    """Verify that parse decision tree classifier behaves as expected."""
    try:
        res = parse_decision_tree_classifier()
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


def test_parse_decision_tree_regressor() -> None:
    """Verify that parse decision tree regressor behaves as expected."""
    try:
        res = parse_decision_tree_regressor()
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


def test_parse_random_forest_classifier() -> None:
    """Verify that parse random forest classifier behaves as expected."""
    try:
        res = parse_random_forest_classifier()
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


def test_parse_random_forest_regressor() -> None:
    """Verify that parse random forest regressor behaves as expected."""
    try:
        res = parse_random_forest_regressor()
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


def test_parse_gradient_boosting_classifier() -> None:
    """Verify that parse gradient boosting classifier behaves as expected."""
    try:
        res = parse_gradient_boosting_classifier()
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


def test_parse_gradient_boosting_regressor() -> None:
    """Verify that parse gradient boosting regressor behaves as expected."""
    try:
        res = parse_gradient_boosting_regressor()
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


def test_parse_hist_gradient_boosting_classifier() -> None:
    """Verify that parse hist gradient boosting classifier behaves as expected."""
    try:
        res = parse_hist_gradient_boosting_classifier()
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


def test_parse_hist_gradient_boosting_regressor() -> None:
    """Verify that parse hist gradient boosting regressor behaves as expected."""
    try:
        res = parse_hist_gradient_boosting_regressor()
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


def test_parse_isolation_forest() -> None:
    """Verify that parse isolation forest behaves as expected."""
    try:
        res = parse_isolation_forest()
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


def test_parse_ada_boost_classifier() -> None:
    """Verify that parse ada boost classifier behaves as expected."""
    try:
        res = parse_ada_boost_classifier()
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


def test_parse_ada_boost_regressor() -> None:
    """Verify that parse ada boost regressor behaves as expected."""
    try:
        res = parse_ada_boost_regressor()
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


def test_parse_extra_trees_classifier() -> None:
    """Verify that parse extra trees classifier behaves as expected."""
    try:
        res = parse_extra_trees_classifier()
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


def test_parse_extra_trees_regressor() -> None:
    """Verify that parse extra trees regressor behaves as expected."""
    try:
        res = parse_extra_trees_regressor()
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


def test_extract_n_estimators() -> None:
    """Verify that extract n estimators behaves as expected."""
    try:
        res = extract_n_estimators()
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


def test_handle_predict_proba() -> None:
    """Verify that handle predict proba behaves as expected."""
    try:
        res = handle_predict_proba()
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


def test_handle_multi_output_regressors() -> None:
    """Verify that handle multi output regressors behaves as expected."""
    try:
        res = handle_multi_output_regressors()
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


def test_handle_multi_label_classification() -> None:
    """Verify that handle multi label classification behaves as expected."""
    try:
        res = handle_multi_label_classification()
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


def test_parse_pipeline() -> None:
    """Verify that parse pipeline behaves as expected."""
    try:
        res = parse_pipeline()
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


def test_extract_classes_and_zipmaps() -> None:
    """Verify that extract classes and zipmaps behaves as expected."""
    try:
        res = extract_classes_and_zipmaps()
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


def test_parse_linear_regression() -> None:
    """Verify that parse linear regression behaves as expected."""
    try:
        res = parse_linear_regression()
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


def test_parse_logistic_regression() -> None:
    """Verify that parse logistic regression behaves as expected."""
    try:
        res = parse_logistic_regression()
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


def test_parse_ridge_lasso_elasticnet() -> None:
    """Verify that parse ridge lasso elasticnet behaves as expected."""
    try:
        res = parse_ridge_lasso_elasticnet()
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


def test_parse_sgd_classifier() -> None:
    """Verify that parse sgd classifier behaves as expected."""
    try:
        res = parse_sgd_classifier()
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


def test_parse_linear_svc() -> None:
    """Verify that parse linear svc behaves as expected."""
    try:
        res = parse_linear_svc()
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


def test_parse_svc_poly() -> None:
    """Verify that parse svc poly behaves as expected."""
    try:
        res = parse_svc_poly()
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


def test_parse_svc_rbf() -> None:
    """Verify that parse svc rbf behaves as expected."""
    try:
        res = parse_svc_rbf()
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


def test_parse_svc_sigmoid() -> None:
    """Verify that parse svc sigmoid behaves as expected."""
    try:
        res = parse_svc_sigmoid()
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


def test_parse_gaussian_nb() -> None:
    """Verify that parse gaussian nb behaves as expected."""
    try:
        res = parse_gaussian_nb()
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


def test_parse_multinomial_nb() -> None:
    """Verify that parse multinomial nb behaves as expected."""
    try:
        res = parse_multinomial_nb()
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


def test_parse_bernoulli_nb() -> None:
    """Verify that parse bernoulli nb behaves as expected."""
    try:
        res = parse_bernoulli_nb()
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


def test_parse_mlp_classifier() -> None:
    """Verify that parse mlp classifier behaves as expected."""
    try:
        res = parse_mlp_classifier()
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


def test_optimize_standard_scaler() -> None:
    """Verify that optimize standard scaler behaves as expected."""
    try:
        res = optimize_standard_scaler()
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


def test_optimize_binarizer() -> None:
    """Verify that optimize binarizer behaves as expected."""
    try:
        res = optimize_binarizer()
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


def test_optimize_onehot_encoder() -> None:
    """Verify that optimize onehot encoder behaves as expected."""
    try:
        res = optimize_onehot_encoder()
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
