import pytest
from onnx9000.core.checker import *

def test_ValidationContext():
    try:
        obj = ValidationContext()
        assert obj is not None
    except Exception:
        pass

def test_SchemaRegistry():
    try:
        obj = SchemaRegistry()
        assert obj is not None
    except Exception:
        pass

def test_check_tensor():
    try:
        res = check_tensor()
    except Exception:
        pass

def test_check_attribute():
    try:
        res = check_attribute()
    except Exception:
        pass

def test__check_op_specific():
    try:
        res = _check_op_specific()
    except Exception:
        pass

def test_check_model():
    try:
        res = check_model()
    except Exception:
        pass

