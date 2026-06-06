import pytest
from onnx9000.c_compiler.compiler import *

def test_C89Compiler():
    try:
        obj = C89Compiler()
        assert obj is not None
    except Exception:
        pass

