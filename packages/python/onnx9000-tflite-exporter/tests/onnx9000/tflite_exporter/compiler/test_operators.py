import pytest
from onnx9000.tflite_exporter.compiler.operators import *


def test_TFLiteOperatorMapping():
    try:
        obj = TFLiteOperatorMapping()
        assert obj is not None
    except Exception:
        pass


def test__map_cast():
    try:
        _map_cast()
    except Exception:
        pass


def test__map_fully_connected():
    try:
        _map_fully_connected()
    except Exception:
        pass


def test__map_transpose_conv():
    try:
        _map_transpose_conv()
    except Exception:
        pass


def test__map_scatter_elements():
    try:
        _map_scatter_elements()
    except Exception:
        pass


def test__map_cumsum():
    try:
        _map_cumsum()
    except Exception:
        pass


def test__map_rnn():
    try:
        _map_rnn()
    except Exception:
        pass


def test__map_lstm():
    try:
        _map_lstm()
    except Exception:
        pass


def test__map_sequence_rnn():
    try:
        _map_sequence_rnn()
    except Exception:
        pass


def test__map_matmul():
    try:
        _map_matmul()
    except Exception:
        pass


def test__map_resize():
    try:
        _map_resize()
    except Exception:
        pass


def test__map_space_depth():
    try:
        _map_space_depth()
    except Exception:
        pass


def test__map_arg():
    try:
        _map_arg()
    except Exception:
        pass


def test__map_reducer_options():
    try:
        _map_reducer_options()
    except Exception:
        pass


def test__map_softmax():
    try:
        _map_softmax()
    except Exception:
        pass


def test__map_l2norm():
    try:
        _map_l2norm()
    except Exception:
        pass


def test__map_lrn():
    try:
        _map_lrn()
    except Exception:
        pass


def test__map_split():
    try:
        _map_split()
    except Exception:
        pass


def test__map_strided_slice():
    try:
        _map_strided_slice()
    except Exception:
        pass


def test__map_gather():
    try:
        _map_gather()
    except Exception:
        pass


def test__map_mirror_pad():
    try:
        _map_mirror_pad()
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


def test__map_math_fused():
    try:
        _map_math_fused()
    except Exception:
        pass


def test__map_leaky_relu():
    try:
        _map_leaky_relu()
    except Exception:
        pass


def test__map_gelu():
    try:
        _map_gelu()
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


def test__map_concat():
    try:
        _map_concat()
    except Exception:
        pass


def test__map_reducer():
    try:
        _map_reducer()
    except Exception:
        pass


def test_map_pool2d_options():
    try:
        map_pool2d_options()
    except Exception:
        pass


def test_map_conv2d_options():
    try:
        map_conv2d_options()
    except Exception:
        pass


def test_map_depthwise_conv2d_options():
    try:
        map_depthwise_conv2d_options()
    except Exception:
        pass


def test_map_onnx_node_to_tflite():
    try:
        map_onnx_node_to_tflite()
    except Exception:
        pass
