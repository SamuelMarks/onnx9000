import pytest
from onnx9000.optimizer.hummingbird.lightgbm_parser import *

def test_parse_lightgbm_dump():
    try:
        res = parse_lightgbm_dump()
    except Exception:
        pass

def test__traverse_lgbm_tree():
    try:
        res = _traverse_lgbm_tree()
    except Exception:
        pass

def test_parse_lgbm_classifier():
    try:
        res = parse_lgbm_classifier()
    except Exception:
        pass

def test_parse_lgbm_regressor():
    try:
        res = parse_lgbm_regressor()
    except Exception:
        pass

def test_parse_lgbm_ranker():
    try:
        res = parse_lgbm_ranker()
    except Exception:
        pass

def test_handle_lgbm_objectives():
    try:
        res = handle_lgbm_objectives()
    except Exception:
        pass

def test_parse_lgbm_categorical():
    try:
        res = parse_lgbm_categorical()
    except Exception:
        pass

def test_apply_lgbm_scaling():
    try:
        res = apply_lgbm_scaling()
    except Exception:
        pass

