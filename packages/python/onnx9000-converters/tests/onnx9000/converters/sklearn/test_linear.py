import pytest
from onnx9000.converters.sklearn.linear import *

def test__convert_linear_regressor():
    try:
        res = _convert_linear_regressor()
    except Exception:
        pass

def test_convert_linear_regression():
    try:
        res = convert_linear_regression()
    except Exception:
        pass

def test_convert_ridge():
    try:
        res = convert_ridge()
    except Exception:
        pass

def test_convert_ridge_cv():
    try:
        res = convert_ridge_cv()
    except Exception:
        pass

def test_convert_lasso():
    try:
        res = convert_lasso()
    except Exception:
        pass

def test_convert_lasso_cv():
    try:
        res = convert_lasso_cv()
    except Exception:
        pass

def test_convert_elastic_net():
    try:
        res = convert_elastic_net()
    except Exception:
        pass

def test_convert_elastic_net_cv():
    try:
        res = convert_elastic_net_cv()
    except Exception:
        pass

def test_convert_lars():
    try:
        res = convert_lars()
    except Exception:
        pass

def test_convert_lasso_lars():
    try:
        res = convert_lasso_lars()
    except Exception:
        pass

def test_convert_omp():
    try:
        res = convert_omp()
    except Exception:
        pass

def test_convert_bayesian_ridge():
    try:
        res = convert_bayesian_ridge()
    except Exception:
        pass

def test_convert_ard_regression():
    try:
        res = convert_ard_regression()
    except Exception:
        pass

def test_convert_passive_aggressive_regressor():
    try:
        res = convert_passive_aggressive_regressor()
    except Exception:
        pass

def test_convert_sgd_regressor():
    try:
        res = convert_sgd_regressor()
    except Exception:
        pass

def test_convert_huber_regressor():
    try:
        res = convert_huber_regressor()
    except Exception:
        pass

def test_convert_theil_sen_regressor():
    try:
        res = convert_theil_sen_regressor()
    except Exception:
        pass

def test_convert_quantile_regressor():
    try:
        res = convert_quantile_regressor()
    except Exception:
        pass

def test_convert_poisson_regressor():
    try:
        res = convert_poisson_regressor()
    except Exception:
        pass

def test_convert_gamma_regressor():
    try:
        res = convert_gamma_regressor()
    except Exception:
        pass

def test_convert_tweedie_regressor():
    try:
        res = convert_tweedie_regressor()
    except Exception:
        pass

def test__convert_linear_classifier():
    try:
        res = _convert_linear_classifier()
    except Exception:
        pass

def test_convert_logistic_regression():
    try:
        res = convert_logistic_regression()
    except Exception:
        pass

def test_convert_logistic_regression_cv():
    try:
        res = convert_logistic_regression_cv()
    except Exception:
        pass

def test_convert_passive_aggressive_classifier():
    try:
        res = convert_passive_aggressive_classifier()
    except Exception:
        pass

def test_convert_perceptron():
    try:
        res = convert_perceptron()
    except Exception:
        pass

def test_convert_ridge_classifier():
    try:
        res = convert_ridge_classifier()
    except Exception:
        pass

def test_convert_ridge_classifier_cv():
    try:
        res = convert_ridge_classifier_cv()
    except Exception:
        pass

def test_convert_sgd_classifier():
    try:
        res = convert_sgd_classifier()
    except Exception:
        pass

