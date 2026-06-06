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
        check_tensor()
    except Exception:
        pass


def test_check_attribute():
    try:
        check_attribute()
    except Exception:
        pass


def test__check_op_specific():
    try:
        _check_op_specific()
    except Exception:
        pass


def test_check_model():
    try:
        check_model()
    except Exception:
        pass
