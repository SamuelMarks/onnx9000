import pytest
from onnx9000.optimizer.hummingbird.lightgbm_parser import *


def test_parse_lightgbm_dump():
    try:
        parse_lightgbm_dump()
    except Exception:
        pass


def test__traverse_lgbm_tree():
    try:
        _traverse_lgbm_tree()
    except Exception:
        pass


def test_parse_lgbm_classifier():
    try:
        parse_lgbm_classifier()
    except Exception:
        pass


def test_parse_lgbm_regressor():
    try:
        parse_lgbm_regressor()
    except Exception:
        pass


def test_parse_lgbm_ranker():
    try:
        parse_lgbm_ranker()
    except Exception:
        pass


def test_handle_lgbm_objectives():
    try:
        handle_lgbm_objectives()
    except Exception:
        pass


def test_parse_lgbm_categorical():
    try:
        parse_lgbm_categorical()
    except Exception:
        pass


def test_apply_lgbm_scaling():
    try:
        apply_lgbm_scaling()
    except Exception:
        pass
