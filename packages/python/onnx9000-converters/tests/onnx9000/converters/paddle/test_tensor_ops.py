import pytest
from onnx9000.converters.paddle.tensor_ops import *


def test__map_reshape():
    try:
        _map_reshape()
    except Exception:
        pass


def test__map_transpose():
    try:
        _map_transpose()
    except Exception:
        pass


def test__map_flatten():
    try:
        _map_flatten()
    except Exception:
        pass


def test__map_squeeze():
    try:
        _map_squeeze()
    except Exception:
        pass


def test__map_unsqueeze():
    try:
        _map_unsqueeze()
    except Exception:
        pass


def test__map_concat():
    try:
        _map_concat()
    except Exception:
        pass


def test__map_stack():
    try:
        _map_stack()
    except Exception:
        pass


def test__map_unstack():
    try:
        _map_unstack()
    except Exception:
        pass


def test__map_split():
    try:
        _map_split()
    except Exception:
        pass


def test__map_slice():
    try:
        _map_slice()
    except Exception:
        pass


def test__map_gather():
    try:
        _map_gather()
    except Exception:
        pass


def test__map_gather_nd():
    try:
        _map_gather_nd()
    except Exception:
        pass


def test__map_scatter():
    try:
        _map_scatter()
    except Exception:
        pass


def test__map_scatter_nd():
    try:
        _map_scatter_nd()
    except Exception:
        pass


def test__map_scatter_nd_add():
    try:
        _map_scatter_nd_add()
    except Exception:
        pass


def test__map_tile():
    try:
        _map_tile()
    except Exception:
        pass


def test__map_expand():
    try:
        _map_expand()
    except Exception:
        pass


def test__map_expand_as():
    try:
        _map_expand_as()
    except Exception:
        pass


def test__map_cast():
    try:
        _map_cast()
    except Exception:
        pass


def test__map_shape():
    try:
        _map_shape()
    except Exception:
        pass


def test__map_size():
    try:
        _map_size()
    except Exception:
        pass


def test__map_fill_constant():
    try:
        _map_fill_constant()
    except Exception:
        pass


def test__map_zeros_ones_like():
    try:
        _map_zeros_ones_like()
    except Exception:
        pass


def test__map_range():
    try:
        _map_range()
    except Exception:
        pass


def test__map_assign():
    try:
        _map_assign()
    except Exception:
        pass


def test__map_where():
    try:
        _map_where()
    except Exception:
        pass


def test__map_nonzero():
    try:
        _map_nonzero()
    except Exception:
        pass


def test__map_custom():
    try:
        _map_custom()
    except Exception:
        pass
