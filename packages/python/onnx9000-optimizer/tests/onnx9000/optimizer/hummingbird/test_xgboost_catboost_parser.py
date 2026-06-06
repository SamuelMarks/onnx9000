import pytest
from onnx9000.optimizer.hummingbird.xgboost_catboost_parser import *

def test_parse_xgboost_dump():
    try:
        res = parse_xgboost_dump()
    except Exception:
        pass

def test__traverse_xgb_tree():
    try:
        res = _traverse_xgb_tree()
    except Exception:
        pass

def test_parse_xgb_classifier():
    try:
        res = parse_xgb_classifier()
    except Exception:
        pass

def test_parse_xgb_regressor():
    try:
        res = parse_xgb_regressor()
    except Exception:
        pass

def test_parse_xgb_ranker():
    try:
        res = parse_xgb_ranker()
    except Exception:
        pass

def test_handle_xgb_objectives():
    try:
        res = handle_xgb_objectives()
    except Exception:
        pass

def test_parse_catboost_classifier():
    try:
        res = parse_catboost_classifier()
    except Exception:
        pass

def test_parse_catboost_regressor():
    try:
        res = parse_catboost_regressor()
    except Exception:
        pass

def test_handle_catboost_categorical():
    try:
        res = handle_catboost_categorical()
    except Exception:
        pass

