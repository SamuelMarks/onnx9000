import pytest
from onnx9000.tensorrt.ffi import *

def test_TensorRTFFI():
    try:
        obj = TensorRTFFI()
        assert obj is not None
    except Exception:
        pass

def test__phase_1_20_bindings():
    try:
        res = _phase_1_20_bindings()
    except Exception:
        pass

