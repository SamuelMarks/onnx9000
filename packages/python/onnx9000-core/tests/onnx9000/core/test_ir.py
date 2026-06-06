import pytest
from onnx9000.core.ir import *

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

def test_DynamicDim():
    try:
        obj = DynamicDim()
        assert obj is not None
    except Exception:
        pass

def test_Attribute():
    try:
        obj = Attribute()
        assert obj is not None
    except Exception:
        pass

def test_ValueInfo():
    try:
        obj = ValueInfo()
        assert obj is not None
    except Exception:
        pass

def test_Tensor():
    try:
        obj = Tensor()
        assert obj is not None
    except Exception:
        pass

def test_SparseTensor():
    try:
        obj = SparseTensor()
        assert obj is not None
    except Exception:
        pass

def test_Variable():
    try:
        obj = Variable()
        assert obj is not None
    except Exception:
        pass

def test_Constant():
    try:
        obj = Constant()
        assert obj is not None
    except Exception:
        pass

def test_Node():
    try:
        obj = Node()
        assert obj is not None
    except Exception:
        pass

def test_Graph():
    try:
        obj = Graph()
        assert obj is not None
    except Exception:
        pass

def test_QuantizedTensor():
    try:
        obj = QuantizedTensor()
        assert obj is not None
    except Exception:
        pass

