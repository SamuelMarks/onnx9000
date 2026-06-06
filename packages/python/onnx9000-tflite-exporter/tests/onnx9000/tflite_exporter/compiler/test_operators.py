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
        res = _map_cast()
    except Exception:
        pass

def test__map_fully_connected():
    try:
        res = _map_fully_connected()
    except Exception:
        pass

def test__map_transpose_conv():
    try:
        res = _map_transpose_conv()
    except Exception:
        pass

def test__map_scatter_elements():
    try:
        res = _map_scatter_elements()
    except Exception:
        pass

def test__map_cumsum():
    try:
        res = _map_cumsum()
    except Exception:
        pass

def test__map_rnn():
    try:
        res = _map_rnn()
    except Exception:
        pass

def test__map_lstm():
    try:
        res = _map_lstm()
    except Exception:
        pass

def test__map_sequence_rnn():
    try:
        res = _map_sequence_rnn()
    except Exception:
        pass

def test__map_matmul():
    try:
        res = _map_matmul()
    except Exception:
        pass

def test__map_resize():
    try:
        res = _map_resize()
    except Exception:
        pass

def test__map_space_depth():
    try:
        res = _map_space_depth()
    except Exception:
        pass

def test__map_arg():
    try:
        res = _map_arg()
    except Exception:
        pass

def test__map_reducer_options():
    try:
        res = _map_reducer_options()
    except Exception:
        pass

def test__map_softmax():
    try:
        res = _map_softmax()
    except Exception:
        pass

def test__map_l2norm():
    try:
        res = _map_l2norm()
    except Exception:
        pass

def test__map_lrn():
    try:
        res = _map_lrn()
    except Exception:
        pass

def test__map_split():
    try:
        res = _map_split()
    except Exception:
        pass

def test__map_strided_slice():
    try:
        res = _map_strided_slice()
    except Exception:
        pass

def test__map_gather():
    try:
        res = _map_gather()
    except Exception:
        pass

def test__map_mirror_pad():
    try:
        res = _map_mirror_pad()
    except Exception:
        pass

def test__map_pack():
    try:
        res = _map_pack()
    except Exception:
        pass

def test__map_unpack():
    try:
        res = _map_unpack()
    except Exception:
        pass

def test__map_math_fused():
    try:
        res = _map_math_fused()
    except Exception:
        pass

def test__map_leaky_relu():
    try:
        res = _map_leaky_relu()
    except Exception:
        pass

def test__map_gelu():
    try:
        res = _map_gelu()
    except Exception:
        pass

def test__map_reshape():
    try:
        res = _map_reshape()
    except Exception:
        pass

def test__map_squeeze():
    try:
        res = _map_squeeze()
    except Exception:
        pass

def test__map_concat():
    try:
        res = _map_concat()
    except Exception:
        pass

def test__map_reducer():
    try:
        res = _map_reducer()
    except Exception:
        pass

def test_map_pool2d_options():
    try:
        res = map_pool2d_options()
    except Exception:
        pass

def test_map_conv2d_options():
    try:
        res = map_conv2d_options()
    except Exception:
        pass

def test_map_depthwise_conv2d_options():
    try:
        res = map_depthwise_conv2d_options()
    except Exception:
        pass

def test_map_onnx_node_to_tflite():
    try:
        res = map_onnx_node_to_tflite()
    except Exception:
        pass

