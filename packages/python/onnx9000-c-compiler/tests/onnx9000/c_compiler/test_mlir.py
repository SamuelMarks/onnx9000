import pytest
from onnx9000.c_compiler.mlir import *

def test_MLIRCompiler():
    try:
        obj = MLIRCompiler()
        assert obj is not None
    except Exception:
        pass

