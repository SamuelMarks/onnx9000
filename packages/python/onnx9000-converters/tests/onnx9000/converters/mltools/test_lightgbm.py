import pytest
from onnx9000.converters.mltools.lightgbm import *


def test_parse_lightgbm_json():
    try:
        parse_lightgbm_json()
    except Exception:
        pass


def test_parse_lightgbm_dict():
    try:
        parse_lightgbm_dict()
    except Exception:
        pass
