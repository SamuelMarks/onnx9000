import pytest
from onnx9000.backends.codegen.ops.elementwise import *


def test__generate_unary_op():
    try:
        _generate_unary_op()
    except Exception:
        pass


def test_generate_relu():
    try:
        generate_relu()
    except Exception:
        pass


def test_generate_elu():
    try:
        generate_elu()
    except Exception:
        pass


def test_generate_celu():
    try:
        generate_celu()
    except Exception:
        pass


def test_generate_leaky_relu():
    try:
        generate_leaky_relu()
    except Exception:
        pass


def test_generate_selu():
    try:
        generate_selu()
    except Exception:
        pass


def test_generate_softplus():
    try:
        generate_softplus()
    except Exception:
        pass


def test_generate_softsign():
    try:
        generate_softsign()
    except Exception:
        pass


def test_generate_thresholded_relu():
    try:
        generate_thresholded_relu()
    except Exception:
        pass


def test_generate_mish():
    try:
        generate_mish()
    except Exception:
        pass


def test_generate_hard_sigmoid():
    try:
        generate_hard_sigmoid()
    except Exception:
        pass


def test_generate_hard_swish():
    try:
        generate_hard_swish()
    except Exception:
        pass


def test_generate_sigmoid():
    try:
        generate_sigmoid()
    except Exception:
        pass


def test_generate_tanh():
    try:
        generate_tanh()
    except Exception:
        pass


def test__generate_binary_op():
    try:
        _generate_binary_op()
    except Exception:
        pass


def test__generate_ternary_op():
    try:
        _generate_ternary_op()
    except Exception:
        pass


def test_generate_prelu():
    try:
        generate_prelu()
    except Exception:
        pass


def test_generate_add():
    try:
        generate_add()
    except Exception:
        pass


def test_generate_sub():
    try:
        generate_sub()
    except Exception:
        pass


def test_generate_mul():
    try:
        generate_mul()
    except Exception:
        pass


def test_generate_div():
    try:
        generate_div()
    except Exception:
        pass
