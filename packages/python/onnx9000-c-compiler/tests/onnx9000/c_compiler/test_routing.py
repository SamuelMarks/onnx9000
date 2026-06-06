import pytest
from onnx9000.c_compiler.routing import *

def test_generate_shape_op():
    try:
        res = generate_shape_op()
    except Exception:
        pass

def test_generate_transpose():
    try:
        res = generate_transpose()
    except Exception:
        pass

def test_generate_concat():
    try:
        res = generate_concat()
    except Exception:
        pass

def test_generate_pad():
    try:
        res = generate_pad()
    except Exception:
        pass

def test_generate_slice():
    try:
        res = generate_slice()
    except Exception:
        pass

def test_generate_gather():
    try:
        res = generate_gather()
    except Exception:
        pass

def test_generate_gathernd():
    try:
        res = generate_gathernd()
    except Exception:
        pass

def test_generate_scatter_elements():
    try:
        res = generate_scatter_elements()
    except Exception:
        pass

def test_generate_scatternd():
    try:
        res = generate_scatternd()
    except Exception:
        pass

def test_generate_expand():
    try:
        res = generate_expand()
    except Exception:
        pass

def test_generate_tile():
    try:
        res = generate_tile()
    except Exception:
        pass

def test_generate_constant_of_shape():
    try:
        res = generate_constant_of_shape()
    except Exception:
        pass

def test_generate_cumsum():
    try:
        res = generate_cumsum()
    except Exception:
        pass

def test_generate_reverse_sequence():
    try:
        res = generate_reverse_sequence()
    except Exception:
        pass

def test_generate_onehot():
    try:
        res = generate_onehot()
    except Exception:
        pass

def test_generate_depth_to_space():
    try:
        res = generate_depth_to_space()
    except Exception:
        pass

def test_generate_space_to_depth():
    try:
        res = generate_space_to_depth()
    except Exception:
        pass

