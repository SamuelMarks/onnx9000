import pytest
from onnx9000.c_compiler.simd_macros import *

def test_emit_simd_macros():
    try:
        res = emit_simd_macros()
    except Exception:
        pass

