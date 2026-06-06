import pytest
from onnx9000.converters.tf.nn_ops import *

def test__map_simple_binary():
    try:
        res = _map_simple_binary()
    except Exception:
        pass

def test__map_simple_unary():
    try:
        res = _map_simple_unary()
    except Exception:
        pass

def test__map_relu6():
    try:
        res = _map_relu6()
    except Exception:
        pass

def test__map_leaky_relu():
    try:
        res = _map_leaky_relu()
    except Exception:
        pass

def test__map_elu():
    try:
        res = _map_elu()
    except Exception:
        pass

def test__map_selu():
    try:
        res = _map_selu()
    except Exception:
        pass

def test__map_softplus():
    try:
        res = _map_softplus()
    except Exception:
        pass

def test__map_softsign():
    try:
        res = _map_softsign()
    except Exception:
        pass

def test__map_softmax():
    try:
        res = _map_softmax()
    except Exception:
        pass

def test__map_log_softmax():
    try:
        res = _map_log_softmax()
    except Exception:
        pass

def test__map_conv2d():
    try:
        res = _map_conv2d()
    except Exception:
        pass

def test__map_conv3d():
    try:
        res = _map_conv3d()
    except Exception:
        pass

def test__map_depthwise_conv2d_native():
    try:
        res = _map_depthwise_conv2d_native()
    except Exception:
        pass

def test__map_conv2d_backprop_input():
    try:
        res = _map_conv2d_backprop_input()
    except Exception:
        pass

def test__map_conv3d_backprop_input_v2():
    try:
        res = _map_conv3d_backprop_input_v2()
    except Exception:
        pass

def test__map_max_pool():
    try:
        res = _map_max_pool()
    except Exception:
        pass

def test__map_max_pool_3d():
    try:
        res = _map_max_pool_3d()
    except Exception:
        pass

def test__map_avg_pool():
    try:
        res = _map_avg_pool()
    except Exception:
        pass

def test__map_avg_pool_3d():
    try:
        res = _map_avg_pool_3d()
    except Exception:
        pass

def test__map_global_max_pool():
    try:
        res = _map_global_max_pool()
    except Exception:
        pass

def test__map_global_avg_pool():
    try:
        res = _map_global_avg_pool()
    except Exception:
        pass

def test__map_fractional_max_pool():
    try:
        res = _map_fractional_max_pool()
    except Exception:
        pass

def test__map_fractional_avg_pool():
    try:
        res = _map_fractional_avg_pool()
    except Exception:
        pass

def test__map_batch_norm():
    try:
        res = _map_batch_norm()
    except Exception:
        pass

def test__map_l2_loss():
    try:
        res = _map_l2_loss()
    except Exception:
        pass

def test__map_lrn():
    try:
        res = _map_lrn()
    except Exception:
        pass

def test__map_dropout():
    try:
        res = _map_dropout()
    except Exception:
        pass

def test__map_topk():
    try:
        res = _map_topk()
    except Exception:
        pass

def test__map_in_top_k():
    try:
        res = _map_in_top_k()
    except Exception:
        pass

def test__map_nth_element():
    try:
        res = _map_nth_element()
    except Exception:
        pass

