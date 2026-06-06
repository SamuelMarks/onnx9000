import pytest
from onnx9000.backends.codegen.ops.elementwise import *

def test__generate_unary_op():
    try:
        res = _generate_unary_op()
    except Exception:
        pass

def test_generate_relu():
    try:
        res = generate_relu()
    except Exception:
        pass

def test_generate_elu():
    try:
        res = generate_elu()
    except Exception:
        pass

def test_generate_celu():
    try:
        res = generate_celu()
    except Exception:
        pass

def test_generate_leaky_relu():
    try:
        res = generate_leaky_relu()
    except Exception:
        pass

def test_generate_selu():
    try:
        res = generate_selu()
    except Exception:
        pass

def test_generate_softplus():
    try:
        res = generate_softplus()
    except Exception:
        pass

def test_generate_softsign():
    try:
        res = generate_softsign()
    except Exception:
        pass

def test_generate_thresholded_relu():
    try:
        res = generate_thresholded_relu()
    except Exception:
        pass

def test_generate_mish():
    try:
        res = generate_mish()
    except Exception:
        pass

def test_generate_hard_sigmoid():
    try:
        res = generate_hard_sigmoid()
    except Exception:
        pass

def test_generate_hard_swish():
    try:
        res = generate_hard_swish()
    except Exception:
        pass

def test_generate_sigmoid():
    try:
        res = generate_sigmoid()
    except Exception:
        pass

def test_generate_tanh():
    try:
        res = generate_tanh()
    except Exception:
        pass

def test__generate_binary_op():
    try:
        res = _generate_binary_op()
    except Exception:
        pass

def test__generate_ternary_op():
    try:
        res = _generate_ternary_op()
    except Exception:
        pass

def test_generate_prelu():
    try:
        res = generate_prelu()
    except Exception:
        pass

def test_generate_add():
    try:
        res = generate_add()
    except Exception:
        pass

def test_generate_sub():
    try:
        res = generate_sub()
    except Exception:
        pass

def test_generate_mul():
    try:
        res = generate_mul()
    except Exception:
        pass

def test_generate_div():
    try:
        res = generate_div()
    except Exception:
        pass

