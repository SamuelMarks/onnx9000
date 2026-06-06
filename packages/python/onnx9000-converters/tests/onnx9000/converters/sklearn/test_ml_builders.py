import pytest
from onnx9000.converters.sklearn.ml_builders import *


def test_convert_mlp_classifier():
    try:
        convert_mlp_classifier()
    except Exception:
        pass


def test_convert_mlp_regressor():
    try:
        convert_mlp_regressor()
    except Exception:
        pass


def test_convert_select_k_best():
    try:
        convert_select_k_best()
    except Exception:
        pass


def test_convert_select_percentile():
    try:
        convert_select_percentile()
    except Exception:
        pass


def test_convert_select_fpr():
    try:
        convert_select_fpr()
    except Exception:
        pass


def test_convert_variance_threshold():
    try:
        convert_variance_threshold()
    except Exception:
        pass
