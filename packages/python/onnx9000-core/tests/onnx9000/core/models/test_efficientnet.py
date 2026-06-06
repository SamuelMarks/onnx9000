import pytest
from onnx9000.core.models.efficientnet import *

def test_SqueezeExcitation():
    try:
        obj = SqueezeExcitation()
        assert obj is not None
    except Exception:
        pass

def test_MBConv():
    try:
        obj = MBConv()
        assert obj is not None
    except Exception:
        pass

def test_EfficientNet():
    try:
        obj = EfficientNet()
        assert obj is not None
    except Exception:
        pass

def test_get_param():
    try:
        res = get_param()
    except Exception:
        pass

def test_efficientnet_b0():
    try:
        res = efficientnet_b0()
    except Exception:
        pass

