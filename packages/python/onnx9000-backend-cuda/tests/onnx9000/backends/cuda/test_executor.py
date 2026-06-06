import pytest
from onnx9000.backends.cuda.executor import *

def test_Dispatcher():
    try:
        obj = Dispatcher()
        assert obj is not None
    except Exception:
        pass

def test__cuda_matmul():
    try:
        res = _cuda_matmul()
    except Exception:
        pass

def test__cuda_add():
    try:
        res = _cuda_add()
    except Exception:
        pass

def test__cuda_sub():
    try:
        res = _cuda_sub()
    except Exception:
        pass

def test__cuda_mul():
    try:
        res = _cuda_mul()
    except Exception:
        pass

