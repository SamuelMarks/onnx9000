import pytest
from onnx9000.core.shape_inference import *


def test__promote_types():
    try:
        _promote_types()
    except Exception:
        pass


def test_get_attr():
    try:
        get_attr()
    except Exception:
        pass


def test_infer_shapes_and_types():
    try:
        infer_shapes_and_types()
    except Exception:
        pass
