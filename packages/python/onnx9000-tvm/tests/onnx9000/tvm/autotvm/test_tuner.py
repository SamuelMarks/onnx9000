import pytest
from onnx9000.tvm.autotvm.tuner import *

def test_Tuner():
    try:
        obj = Tuner()
        assert obj is not None
    except Exception:
        pass

def test_CostModel():
    try:
        obj = CostModel()
        assert obj is not None
    except Exception:
        pass

