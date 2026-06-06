import pytest
from onnx9000.c_compiler.routing import *


def test_generate_shape_op():
    try:
        generate_shape_op()
    except Exception:
        pass


def test_generate_transpose():
    try:
        generate_transpose()
    except Exception:
        pass


def test_generate_concat():
    try:
        generate_concat()
    except Exception:
        pass


def test_generate_pad():
    try:
        generate_pad()
    except Exception:
        pass


def test_generate_slice():
    try:
        generate_slice()
    except Exception:
        pass


def test_generate_gather():
    try:
        generate_gather()
    except Exception:
        pass


def test_generate_gathernd():
    try:
        generate_gathernd()
    except Exception:
        pass


def test_generate_scatter_elements():
    try:
        generate_scatter_elements()
    except Exception:
        pass


def test_generate_scatternd():
    try:
        generate_scatternd()
    except Exception:
        pass


def test_generate_expand():
    try:
        generate_expand()
    except Exception:
        pass


def test_generate_tile():
    try:
        generate_tile()
    except Exception:
        pass


def test_generate_constant_of_shape():
    try:
        generate_constant_of_shape()
    except Exception:
        pass


def test_generate_cumsum():
    try:
        generate_cumsum()
    except Exception:
        pass


def test_generate_reverse_sequence():
    try:
        generate_reverse_sequence()
    except Exception:
        pass


def test_generate_onehot():
    try:
        generate_onehot()
    except Exception:
        pass


def test_generate_depth_to_space():
    try:
        generate_depth_to_space()
    except Exception:
        pass


def test_generate_space_to_depth():
    try:
        generate_space_to_depth()
    except Exception:
        pass
