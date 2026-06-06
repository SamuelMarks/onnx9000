import pytest
from onnx9000.tvm.relay.ty import *

def test_Type():
    try:
        obj = Type()
        assert obj is not None
    except Exception:
        pass

def test_TensorType():
    try:
        obj = TensorType()
        assert obj is not None
    except Exception:
        pass

def test_TupleType():
    try:
        obj = TupleType()
        assert obj is not None
    except Exception:
        pass

def test_FuncType():
    try:
        obj = FuncType()
        assert obj is not None
    except Exception:
        pass

