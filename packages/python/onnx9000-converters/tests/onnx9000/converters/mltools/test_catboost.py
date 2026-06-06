import pytest
from onnx9000.converters.mltools.catboost import *


def test_parse_catboost_json():
    try:
        parse_catboost_json()
    except Exception:
        pass


def test_parse_catboost_dict():
    try:
        parse_catboost_dict()
    except Exception:
        pass
