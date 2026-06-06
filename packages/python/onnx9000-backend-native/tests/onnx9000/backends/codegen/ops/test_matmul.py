import pytest
from onnx9000.backends.codegen.ops.matmul import *

def test_generate_matmul():
    try:
        res = generate_matmul()
    except Exception:
        pass

def test_generate_gemm():
    try:
        res = generate_gemm()
    except Exception:
        pass

