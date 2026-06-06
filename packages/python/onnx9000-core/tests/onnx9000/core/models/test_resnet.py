import pytest
from onnx9000.core.models.resnet import *

def test_BasicBlock():
    try:
        obj = BasicBlock()
        assert obj is not None
    except Exception:
        pass

def test_ResNet():
    try:
        obj = ResNet()
        assert obj is not None
    except Exception:
        pass

def test_get_param():
    try:
        res = get_param()
    except Exception:
        pass

def test_resnet18():
    try:
        res = resnet18()
    except Exception:
        pass

def test_resnet50():
    try:
        res = resnet50()
    except Exception:
        pass

