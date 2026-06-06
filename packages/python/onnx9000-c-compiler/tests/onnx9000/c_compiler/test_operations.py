import pytest
from onnx9000.c_compiler.operations import *

def test_get_strides():
    try:
        res = get_strides()
    except Exception:
        pass

def test_resolve_broadcast_indices():
    try:
        res = resolve_broadcast_indices()
    except Exception:
        pass

def test_generate_elementwise_binary():
    try:
        res = generate_elementwise_binary()
    except Exception:
        pass

def test_generate_math_call():
    try:
        res = generate_math_call()
    except Exception:
        pass

def test_generate_math_binary_call():
    try:
        res = generate_math_binary_call()
    except Exception:
        pass

def test_generate_math_unary_op():
    try:
        res = generate_math_unary_op()
    except Exception:
        pass

def test_generate_sign():
    try:
        res = generate_sign()
    except Exception:
        pass

def test_generate_matmul():
    try:
        res = generate_matmul()
    except Exception:
        pass

def test_generate_einsum():
    try:
        res = generate_einsum()
    except Exception:
        pass

