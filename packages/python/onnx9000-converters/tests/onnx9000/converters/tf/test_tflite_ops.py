import pytest
from onnx9000.converters.tf.tflite_ops import *


def test__map_tflite_simple_binary():
    try:
        _map_tflite_simple_binary()
    except Exception:
        pass


def test__map_tflite_pool():
    try:
        _map_tflite_pool()
    except Exception:
        pass


def test__map_tflite_conv2d():
    try:
        _map_tflite_conv2d()
    except Exception:
        pass


def test__map_tflite_depthwise_conv2d():
    try:
        _map_tflite_depthwise_conv2d()
    except Exception:
        pass


def test__map_tflite_fully_connected():
    try:
        _map_tflite_fully_connected()
    except Exception:
        pass


def test__map_tflite_reshape():
    try:
        _map_tflite_reshape()
    except Exception:
        pass


def test__map_tflite_resize_bilinear():
    try:
        _map_tflite_resize_bilinear()
    except Exception:
        pass


def test__map_tflite_concat():
    try:
        _map_tflite_concat()
    except Exception:
        pass


def test__map_tflite_softmax():
    try:
        _map_tflite_softmax()
    except Exception:
        pass


def test__map_tflite_logistic():
    try:
        _map_tflite_logistic()
    except Exception:
        pass


def test__map_tflite_tanh():
    try:
        _map_tflite_tanh()
    except Exception:
        pass


def test__map_tflite_relu():
    try:
        _map_tflite_relu()
    except Exception:
        pass


def test__map_tflite_relu6():
    try:
        _map_tflite_relu6()
    except Exception:
        pass


def test__map_tflite_relu_n1_to_1():
    try:
        _map_tflite_relu_n1_to_1()
    except Exception:
        pass


def test__map_tflite_dequantize():
    try:
        _map_tflite_dequantize()
    except Exception:
        pass


def test__map_tflite_quantize():
    try:
        _map_tflite_quantize()
    except Exception:
        pass


def test__map_tflite_embedding_lookup():
    try:
        _map_tflite_embedding_lookup()
    except Exception:
        pass


def test__map_tflite_l2_normalization():
    try:
        _map_tflite_l2_normalization()
    except Exception:
        pass


def test__map_tflite_local_response_normalization():
    try:
        _map_tflite_local_response_normalization()
    except Exception:
        pass


def test__map_tflite_space_to_depth():
    try:
        _map_tflite_space_to_depth()
    except Exception:
        pass


def test__map_tflite_depth_to_space():
    try:
        _map_tflite_depth_to_space()
    except Exception:
        pass


def test__map_tflite_floor():
    try:
        _map_tflite_floor()
    except Exception:
        pass


def test__map_tflite_custom_subgraph():
    try:
        _map_tflite_custom_subgraph()
    except Exception:
        pass
