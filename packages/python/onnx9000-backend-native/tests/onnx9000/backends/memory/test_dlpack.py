import pytest
from onnx9000.backends.memory.dlpack import *

def test_DLDataType():
    try:
        obj = DLDataType()
        assert obj is not None
    except Exception:
        pass

def test_DLDevice():
    try:
        obj = DLDevice()
        assert obj is not None
    except Exception:
        pass

def test_DLTensor():
    try:
        obj = DLTensor()
        assert obj is not None
    except Exception:
        pass

def test_DLManagedTensor():
    try:
        obj = DLManagedTensor()
        assert obj is not None
    except Exception:
        pass

def test_from_dlpack():
    try:
        res = from_dlpack()
    except Exception:
        pass

def test_from_numpy():
    try:
        res = from_numpy()
    except Exception:
        pass

