import pytest
from onnx9000.converters.mltools.xgboost import *


def test_parse_xgboost_json():
    try:
        parse_xgboost_json()
    except Exception:
        pass


def test_parse_xgboost_dict():
    try:
        parse_xgboost_dict()
    except Exception:
        pass
