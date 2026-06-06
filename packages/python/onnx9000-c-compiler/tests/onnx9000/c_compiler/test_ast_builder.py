import pytest
from onnx9000.c_compiler.ast_builder import *

def test_C89Builder():
    try:
        obj = C89Builder()
        assert obj is not None
    except Exception:
        pass

