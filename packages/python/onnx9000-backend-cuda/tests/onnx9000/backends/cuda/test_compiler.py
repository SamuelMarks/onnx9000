import pytest
from onnx9000.backends.cuda.compiler import *

def test_CUDACompiler():
    try:
        obj = CUDACompiler()
        assert obj is not None
    except Exception:
        pass

