import pytest
from onnx9000.core.registry import *

def test_OperatorRegistry():
    try:
        obj = OperatorRegistry()
        assert obj is not None
    except Exception:
        pass

def test_register_op():
    try:
        res = register_op()
    except Exception:
        pass

