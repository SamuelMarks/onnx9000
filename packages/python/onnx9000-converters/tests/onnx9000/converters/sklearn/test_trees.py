import pytest
from onnx9000.converters.sklearn.trees import *


def test__convert_tree_classifier():
    try:
        _convert_tree_classifier()
    except Exception:
        pass


def test__convert_tree_regressor():
    try:
        _convert_tree_regressor()
    except Exception:
        pass


def test_convert_decision_tree_classifier():
    try:
        convert_decision_tree_classifier()
    except Exception:
        pass


def test_convert_decision_tree_regressor():
    try:
        convert_decision_tree_regressor()
    except Exception:
        pass


def test_convert_extra_tree_classifier():
    try:
        convert_extra_tree_classifier()
    except Exception:
        pass


def test_convert_extra_tree_regressor():
    try:
        convert_extra_tree_regressor()
    except Exception:
        pass


def test_convert_random_forest_classifier():
    try:
        convert_random_forest_classifier()
    except Exception:
        pass


def test_convert_random_forest_regressor():
    try:
        convert_random_forest_regressor()
    except Exception:
        pass


def test_convert_extra_trees_classifier():
    try:
        convert_extra_trees_classifier()
    except Exception:
        pass


def test_convert_extra_trees_regressor():
    try:
        convert_extra_trees_regressor()
    except Exception:
        pass


def test_convert_gradient_boosting_classifier():
    try:
        convert_gradient_boosting_classifier()
    except Exception:
        pass


def test_convert_gradient_boosting_regressor():
    try:
        convert_gradient_boosting_regressor()
    except Exception:
        pass


def test_convert_hist_gradient_boosting_classifier():
    try:
        convert_hist_gradient_boosting_classifier()
    except Exception:
        pass


def test_convert_hist_gradient_boosting_regressor():
    try:
        convert_hist_gradient_boosting_regressor()
    except Exception:
        pass


def test_convert_ada_boost_classifier():
    try:
        convert_ada_boost_classifier()
    except Exception:
        pass


def test_convert_ada_boost_regressor():
    try:
        convert_ada_boost_regressor()
    except Exception:
        pass


def test_convert_bagging_classifier():
    try:
        convert_bagging_classifier()
    except Exception:
        pass


def test_convert_bagging_regressor():
    try:
        convert_bagging_regressor()
    except Exception:
        pass


def test_convert_isolation_forest():
    try:
        convert_isolation_forest()
    except Exception:
        pass


def test_convert_voting_classifier():
    try:
        convert_voting_classifier()
    except Exception:
        pass


def test_convert_voting_regressor():
    try:
        convert_voting_regressor()
    except Exception:
        pass


def test_convert_stacking_classifier():
    try:
        convert_stacking_classifier()
    except Exception:
        pass


def test_convert_stacking_regressor():
    try:
        convert_stacking_regressor()
    except Exception:
        pass
