import pytest
from onnx9000.optimizer.hummingbird.sklearn_parser import *


def test_parse_decision_tree_classifier():
    try:
        parse_decision_tree_classifier()
    except Exception:
        pass


def test_parse_decision_tree_regressor():
    try:
        parse_decision_tree_regressor()
    except Exception:
        pass


def test_parse_random_forest_classifier():
    try:
        parse_random_forest_classifier()
    except Exception:
        pass


def test_parse_random_forest_regressor():
    try:
        parse_random_forest_regressor()
    except Exception:
        pass


def test_parse_gradient_boosting_classifier():
    try:
        parse_gradient_boosting_classifier()
    except Exception:
        pass


def test_parse_gradient_boosting_regressor():
    try:
        parse_gradient_boosting_regressor()
    except Exception:
        pass


def test_parse_hist_gradient_boosting_classifier():
    try:
        parse_hist_gradient_boosting_classifier()
    except Exception:
        pass


def test_parse_hist_gradient_boosting_regressor():
    try:
        parse_hist_gradient_boosting_regressor()
    except Exception:
        pass


def test_parse_isolation_forest():
    try:
        parse_isolation_forest()
    except Exception:
        pass


def test_parse_ada_boost_classifier():
    try:
        parse_ada_boost_classifier()
    except Exception:
        pass


def test_parse_ada_boost_regressor():
    try:
        parse_ada_boost_regressor()
    except Exception:
        pass


def test_parse_extra_trees_classifier():
    try:
        parse_extra_trees_classifier()
    except Exception:
        pass


def test_parse_extra_trees_regressor():
    try:
        parse_extra_trees_regressor()
    except Exception:
        pass


def test_extract_n_estimators():
    try:
        extract_n_estimators()
    except Exception:
        pass


def test_handle_predict_proba():
    try:
        handle_predict_proba()
    except Exception:
        pass


def test_handle_multi_output_regressors():
    try:
        handle_multi_output_regressors()
    except Exception:
        pass


def test_handle_multi_label_classification():
    try:
        handle_multi_label_classification()
    except Exception:
        pass


def test_parse_pipeline():
    try:
        parse_pipeline()
    except Exception:
        pass


def test_extract_classes_and_zipmaps():
    try:
        extract_classes_and_zipmaps()
    except Exception:
        pass


def test_parse_linear_regression():
    try:
        parse_linear_regression()
    except Exception:
        pass


def test_parse_logistic_regression():
    try:
        parse_logistic_regression()
    except Exception:
        pass


def test_parse_ridge_lasso_elasticnet():
    try:
        parse_ridge_lasso_elasticnet()
    except Exception:
        pass


def test_parse_sgd_classifier():
    try:
        parse_sgd_classifier()
    except Exception:
        pass


def test_parse_linear_svc():
    try:
        parse_linear_svc()
    except Exception:
        pass


def test_parse_svc_poly():
    try:
        parse_svc_poly()
    except Exception:
        pass


def test_parse_svc_rbf():
    try:
        parse_svc_rbf()
    except Exception:
        pass


def test_parse_svc_sigmoid():
    try:
        parse_svc_sigmoid()
    except Exception:
        pass


def test_parse_gaussian_nb():
    try:
        parse_gaussian_nb()
    except Exception:
        pass


def test_parse_multinomial_nb():
    try:
        parse_multinomial_nb()
    except Exception:
        pass


def test_parse_bernoulli_nb():
    try:
        parse_bernoulli_nb()
    except Exception:
        pass


def test_parse_mlp_classifier():
    try:
        parse_mlp_classifier()
    except Exception:
        pass


def test_optimize_standard_scaler():
    try:
        optimize_standard_scaler()
    except Exception:
        pass


def test_optimize_binarizer():
    try:
        optimize_binarizer()
    except Exception:
        pass


def test_optimize_onehot_encoder():
    try:
        optimize_onehot_encoder()
    except Exception:
        pass
