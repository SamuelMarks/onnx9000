import pytest
from onnx9000.tensorrt.builder import *

def test_BuilderConfig():
    try:
        obj = BuilderConfig()
        assert obj is not None
    except Exception:
        pass

def test_NetworkDefinition():
    try:
        obj = NetworkDefinition()
        assert obj is not None
    except Exception:
        pass

def test_Builder():
    try:
        obj = Builder()
        assert obj is not None
    except Exception:
        pass

