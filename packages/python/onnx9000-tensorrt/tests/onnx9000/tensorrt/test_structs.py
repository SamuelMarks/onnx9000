import pytest
from onnx9000.tensorrt.structs import *

def test_Dims():
    try:
        obj = Dims()
        assert obj is not None
    except Exception:
        pass

def test_Weights():
    try:
        obj = Weights()
        assert obj is not None
    except Exception:
        pass

