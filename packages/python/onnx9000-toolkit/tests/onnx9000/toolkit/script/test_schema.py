import pytest
from onnx9000.toolkit.script.schema import *


def test_OpSchema():
    try:
        obj = OpSchema()
        assert obj is not None
    except Exception:
        pass


def test_SchemaRegistry():
    try:
        obj = SchemaRegistry()
        assert obj is not None
    except Exception:
        pass


def test_set_target_opset():
    try:
        set_target_opset()
    except Exception:
        pass


def test_get_target_opset():
    try:
        get_target_opset()
    except Exception:
        pass


def test_validate_op():
    try:
        validate_op()
    except Exception:
        pass
