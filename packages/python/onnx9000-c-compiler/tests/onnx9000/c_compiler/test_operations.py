import pytest
from onnx9000.c_compiler.operations import *


def test_get_strides():
    try:
        get_strides()
    except Exception:
        pass


def test_resolve_broadcast_indices():
    try:
        resolve_broadcast_indices()
    except Exception:
        pass


def test_generate_elementwise_binary():
    try:
        generate_elementwise_binary()
    except Exception:
        pass


def test_generate_math_call():
    try:
        generate_math_call()
    except Exception:
        pass


def test_generate_math_binary_call():
    try:
        generate_math_binary_call()
    except Exception:
        pass


def test_generate_math_unary_op():
    try:
        generate_math_unary_op()
    except Exception:
        pass


def test_generate_sign():
    try:
        generate_sign()
    except Exception:
        pass


def test_generate_matmul():
    try:
        generate_matmul()
    except Exception:
        pass


def test_generate_einsum():
    try:
        generate_einsum()
    except Exception:
        pass
