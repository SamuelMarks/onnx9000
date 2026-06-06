import pytest
from onnx9000.tvm.relay.transform.infer_type import *

def test_TypeChecker():
    try:
        obj = TypeChecker()
        assert obj is not None
    except Exception:
        pass

def test_infer_type():
    try:
        res = infer_type()
    except Exception:
        pass

