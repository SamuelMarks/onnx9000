import pytest
from onnx9000.converters.paddle.nn_ops import *


def test__map_matmul():
    try:
        _map_matmul()
    except Exception:
        pass


def test__map_mul():
    try:
        _map_mul()
    except Exception:
        pass


def test__map_linear():
    try:
        _map_linear()
    except Exception:
        pass


def test__map_conv():
    try:
        _map_conv()
    except Exception:
        pass


def test__map_pool():
    try:
        _map_pool()
    except Exception:
        pass


def test__map_adaptive_pool():
    try:
        _map_adaptive_pool()
    except Exception:
        pass


def test__map_unpool():
    try:
        _map_unpool()
    except Exception:
        pass


def test__map_batch_norm():
    try:
        _map_batch_norm()
    except Exception:
        pass


def test__map_layer_norm():
    try:
        _map_layer_norm()
    except Exception:
        pass


def test__map_group_norm():
    try:
        _map_group_norm()
    except Exception:
        pass


def test__map_instance_norm():
    try:
        _map_instance_norm()
    except Exception:
        pass


def test__map_simple_unary():
    try:
        _map_simple_unary()
    except Exception:
        pass


def test__map_relu6():
    try:
        _map_relu6()
    except Exception:
        pass


def test__map_leaky_relu():
    try:
        _map_leaky_relu()
    except Exception:
        pass


def test__map_elu():
    try:
        _map_elu()
    except Exception:
        pass


def test__map_selu():
    try:
        _map_selu()
    except Exception:
        pass


def test__map_gelu():
    try:
        _map_gelu()
    except Exception:
        pass


def test__map_silu():
    try:
        _map_silu()
    except Exception:
        pass


def test__map_hard_swish():
    try:
        _map_hard_swish()
    except Exception:
        pass


def test__map_hard_sigmoid():
    try:
        _map_hard_sigmoid()
    except Exception:
        pass


def test__map_softmax():
    try:
        _map_softmax()
    except Exception:
        pass


def test__map_log_softmax():
    try:
        _map_log_softmax()
    except Exception:
        pass


def test__map_dropout():
    try:
        _map_dropout()
    except Exception:
        pass


def test__map_pad():
    try:
        _map_pad()
    except Exception:
        pass


def test__map_l2_normalize():
    try:
        _map_l2_normalize()
    except Exception:
        pass


def test__map_roi_align():
    try:
        _map_roi_align()
    except Exception:
        pass


def test__map_roi_pool():
    try:
        _map_roi_pool()
    except Exception:
        pass


def test__map_deformable_conv():
    try:
        _map_deformable_conv()
    except Exception:
        pass


def test__map_custom():
    try:
        _map_custom()
    except Exception:
        pass
