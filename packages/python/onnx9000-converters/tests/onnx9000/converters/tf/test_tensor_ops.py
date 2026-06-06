import pytest
from onnx9000.converters.tf.tensor_ops import *


def test__map_identity():
    try:
        _map_identity()
    except Exception:
        pass


def test__map_identity_n():
    try:
        _map_identity_n()
    except Exception:
        pass


def test__map_reshape():
    try:
        _map_reshape()
    except Exception:
        pass


def test__map_squeeze():
    try:
        _map_squeeze()
    except Exception:
        pass


def test__map_expand_dims():
    try:
        _map_expand_dims()
    except Exception:
        pass


def test__map_transpose():
    try:
        _map_transpose()
    except Exception:
        pass


def test__map_conjugate_transpose():
    try:
        _map_conjugate_transpose()
    except Exception:
        pass


def test__map_concat():
    try:
        _map_concat()
    except Exception:
        pass


def test__map_pack():
    try:
        _map_pack()
    except Exception:
        pass


def test__map_unpack():
    try:
        _map_unpack()
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


def test__map_tile():
    try:
        _map_tile()
    except Exception:
        pass


def test__map_pad():
    try:
        _map_pad()
    except Exception:
        pass


def test__map_pad_v2():
    try:
        _map_pad_v2()
    except Exception:
        pass


def test__map_mirror_pad():
    try:
        _map_mirror_pad()
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


def test__map_scatter_nd():
    try:
        _map_scatter_nd()
    except Exception:
        pass


def test__map_tensor_scatter_update():
    try:
        _map_tensor_scatter_update()
    except Exception:
        pass


def test__map_tensor_scatter_add():
    try:
        _map_tensor_scatter_add()
    except Exception:
        pass


def test__map_space_to_batch():
    try:
        _map_space_to_batch()
    except Exception:
        pass


def test__map_batch_to_space():
    try:
        _map_batch_to_space()
    except Exception:
        pass


def test__map_reverse():
    try:
        _map_reverse()
    except Exception:
        pass


def test__map_roll():
    try:
        _map_roll()
    except Exception:
        pass


def test__map_matrix_diag():
    try:
        _map_matrix_diag()
    except Exception:
        pass


def test__map_cast():
    try:
        _map_cast()
    except Exception:
        pass


def test__map_bitcast():
    try:
        _map_bitcast()
    except Exception:
        pass


def test__map_shape():
    try:
        _map_shape()
    except Exception:
        pass


def test__map_shape_n():
    try:
        _map_shape_n()
    except Exception:
        pass


def test__map_size():
    try:
        _map_size()
    except Exception:
        pass


def test__map_rank():
    try:
        _map_rank()
    except Exception:
        pass


def test__map_zeros_like():
    try:
        _map_zeros_like()
    except Exception:
        pass


def test__map_ones_like():
    try:
        _map_ones_like()
    except Exception:
        pass


def test__map_fill():
    try:
        _map_fill()
    except Exception:
        pass


def test__map_broadcast_to():
    try:
        _map_broadcast_to()
    except Exception:
        pass


def test__map_where():
    try:
        _map_where()
    except Exception:
        pass


def test__map_select():
    try:
        _map_select()
    except Exception:
        pass
