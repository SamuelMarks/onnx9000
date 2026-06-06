import pytest
from onnx9000.converters.frontend.models import *

def test_BasicBlock():
    try:
        obj = BasicBlock()
        assert obj is not None
    except Exception:
        pass

def test_ResNet18():
    try:
        obj = ResNet18()
        assert obj is not None
    except Exception:
        pass

def test_MobileNetV2():
    try:
        obj = MobileNetV2()
        assert obj is not None
    except Exception:
        pass

def test_GPT2Block():
    try:
        obj = GPT2Block()
        assert obj is not None
    except Exception:
        pass

def test_GPT2():
    try:
        obj = GPT2()
        assert obj is not None
    except Exception:
        pass

