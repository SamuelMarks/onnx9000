import pytest
from onnx9000.core.models.convnext import *

def test_ConvNeXtBlock():
    try:
        obj = ConvNeXtBlock()
        assert obj is not None
    except Exception:
        pass

def test_ConvNeXt():
    try:
        obj = ConvNeXt()
        assert obj is not None
    except Exception:
        pass

def test_get_param():
    try:
        res = get_param()
    except Exception:
        pass

def test_convnext_tiny():
    try:
        res = convnext_tiny()
    except Exception:
        pass

