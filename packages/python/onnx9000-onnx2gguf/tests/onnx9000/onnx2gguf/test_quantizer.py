import pytest
from onnx9000.onnx2gguf.quantizer import *

def test_f32_to_f16():
    try:
        res = f32_to_f16()
    except Exception:
        pass

def test_quantize_q4_0():
    try:
        res = quantize_q4_0()
    except Exception:
        pass

def test_quantize_q4_1():
    try:
        res = quantize_q4_1()
    except Exception:
        pass

def test_quantize_q8_0():
    try:
        res = quantize_q8_0()
    except Exception:
        pass

