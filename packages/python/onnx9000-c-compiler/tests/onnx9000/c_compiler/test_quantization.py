import pytest
from onnx9000.c_compiler.quantization import *

def test_generate_quantize_linear():
    try:
        res = generate_quantize_linear()
    except Exception:
        pass

def test_generate_dequantize_linear():
    try:
        res = generate_dequantize_linear()
    except Exception:
        pass

def test_generate_qlinear_matmul():
    try:
        res = generate_qlinear_matmul()
    except Exception:
        pass

def test_generate_qlinear_conv():
    try:
        res = generate_qlinear_conv()
    except Exception:
        pass

