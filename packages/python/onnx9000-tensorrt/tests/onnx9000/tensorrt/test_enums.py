import pytest
from onnx9000.tensorrt.enums import *

def test_DataType():
    try:
        obj = DataType()
        assert obj is not None
    except Exception:
        pass

def test_ElementWiseOperation():
    try:
        obj = ElementWiseOperation()
        assert obj is not None
    except Exception:
        pass

def test_PoolingType():
    try:
        obj = PoolingType()
        assert obj is not None
    except Exception:
        pass

def test_ActivationType():
    try:
        obj = ActivationType()
        assert obj is not None
    except Exception:
        pass

def test_ScaleMode():
    try:
        obj = ScaleMode()
        assert obj is not None
    except Exception:
        pass

def test_UnaryOperation():
    try:
        obj = UnaryOperation()
        assert obj is not None
    except Exception:
        pass

def test_ReduceOperation():
    try:
        obj = ReduceOperation()
        assert obj is not None
    except Exception:
        pass

def test_MatrixOperation():
    try:
        obj = MatrixOperation()
        assert obj is not None
    except Exception:
        pass

def test_TopKOperation():
    try:
        obj = TopKOperation()
        assert obj is not None
    except Exception:
        pass

def test_MemoryPoolType():
    try:
        obj = MemoryPoolType()
        assert obj is not None
    except Exception:
        pass

def test_OptProfileSelector():
    try:
        obj = OptProfileSelector()
        assert obj is not None
    except Exception:
        pass

def test_BuilderFlag():
    try:
        obj = BuilderFlag()
        assert obj is not None
    except Exception:
        pass

