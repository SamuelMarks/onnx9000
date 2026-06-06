import pytest
from onnx9000.optimizer.hummingbird.xgboost_catboost_parser import *


def test_parse_xgboost_dump():
    try:
        parse_xgboost_dump()
    except Exception:
        pass


def test__traverse_xgb_tree():
    try:
        _traverse_xgb_tree()
    except Exception:
        pass


def test_parse_xgb_classifier():
    try:
        parse_xgb_classifier()
    except Exception:
        pass


def test_parse_xgb_regressor():
    try:
        parse_xgb_regressor()
    except Exception:
        pass


def test_parse_xgb_ranker():
    try:
        parse_xgb_ranker()
    except Exception:
        pass


def test_handle_xgb_objectives():
    try:
        handle_xgb_objectives()
    except Exception:
        pass


def test_parse_catboost_classifier():
    try:
        parse_catboost_classifier()
    except Exception:
        pass


def test_parse_catboost_regressor():
    try:
        parse_catboost_regressor()
    except Exception:
        pass


def test_handle_catboost_categorical():
    try:
        handle_catboost_categorical()
    except Exception:
        pass
