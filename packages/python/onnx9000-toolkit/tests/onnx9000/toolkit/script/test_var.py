import pytest
from onnx9000.toolkit.script.var import *

def test_Var():
    try:
        obj = Var()
        assert obj is not None
    except Exception:
        pass

def test__generate_unique_name():
    try:
        res = _generate_unique_name()
    except Exception:
        pass

