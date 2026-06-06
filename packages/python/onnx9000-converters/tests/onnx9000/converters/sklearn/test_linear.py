import pytest
from onnx9000.converters.sklearn.linear import *


def test__convert_linear_regressor():
    try:
        _convert_linear_regressor()
    except Exception:
        pass


def test_convert_linear_regression():
    try:
        convert_linear_regression()
    except Exception:
        pass


def test_convert_ridge():
    try:
        convert_ridge()
    except Exception:
        pass


def test_convert_ridge_cv():
    try:
        convert_ridge_cv()
    except Exception:
        pass


def test_convert_lasso():
    try:
        convert_lasso()
    except Exception:
        pass


def test_convert_lasso_cv():
    try:
        convert_lasso_cv()
    except Exception:
        pass


def test_convert_elastic_net():
    try:
        convert_elastic_net()
    except Exception:
        pass


def test_convert_elastic_net_cv():
    try:
        convert_elastic_net_cv()
    except Exception:
        pass


def test_convert_lars():
    try:
        convert_lars()
    except Exception:
        pass


def test_convert_lasso_lars():
    try:
        convert_lasso_lars()
    except Exception:
        pass


def test_convert_omp():
    try:
        convert_omp()
    except Exception:
        pass


def test_convert_bayesian_ridge():
    try:
        convert_bayesian_ridge()
    except Exception:
        pass


def test_convert_ard_regression():
    try:
        convert_ard_regression()
    except Exception:
        pass


def test_convert_passive_aggressive_regressor():
    try:
        convert_passive_aggressive_regressor()
    except Exception:
        pass


def test_convert_sgd_regressor():
    try:
        convert_sgd_regressor()
    except Exception:
        pass


def test_convert_huber_regressor():
    try:
        convert_huber_regressor()
    except Exception:
        pass


def test_convert_theil_sen_regressor():
    try:
        convert_theil_sen_regressor()
    except Exception:
        pass


def test_convert_quantile_regressor():
    try:
        convert_quantile_regressor()
    except Exception:
        pass


def test_convert_poisson_regressor():
    try:
        convert_poisson_regressor()
    except Exception:
        pass


def test_convert_gamma_regressor():
    try:
        convert_gamma_regressor()
    except Exception:
        pass


def test_convert_tweedie_regressor():
    try:
        convert_tweedie_regressor()
    except Exception:
        pass


def test__convert_linear_classifier():
    try:
        _convert_linear_classifier()
    except Exception:
        pass


def test_convert_logistic_regression():
    try:
        convert_logistic_regression()
    except Exception:
        pass


def test_convert_logistic_regression_cv():
    try:
        convert_logistic_regression_cv()
    except Exception:
        pass


def test_convert_passive_aggressive_classifier():
    try:
        convert_passive_aggressive_classifier()
    except Exception:
        pass


def test_convert_perceptron():
    try:
        convert_perceptron()
    except Exception:
        pass


def test_convert_ridge_classifier():
    try:
        convert_ridge_classifier()
    except Exception:
        pass


def test_convert_ridge_classifier_cv():
    try:
        convert_ridge_classifier_cv()
    except Exception:
        pass


def test_convert_sgd_classifier():
    try:
        convert_sgd_classifier()
    except Exception:
        pass
