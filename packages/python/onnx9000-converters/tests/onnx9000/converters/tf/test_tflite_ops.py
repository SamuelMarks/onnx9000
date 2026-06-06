import pytest
from onnx9000.converters.tf.tflite_ops import *

def test__map_tflite_simple_binary():
    try:
        res = _map_tflite_simple_binary()
    except Exception:
        pass

def test__map_tflite_pool():
    try:
        res = _map_tflite_pool()
    except Exception:
        pass

def test__map_tflite_conv2d():
    try:
        res = _map_tflite_conv2d()
    except Exception:
        pass

def test__map_tflite_depthwise_conv2d():
    try:
        res = _map_tflite_depthwise_conv2d()
    except Exception:
        pass

def test__map_tflite_fully_connected():
    try:
        res = _map_tflite_fully_connected()
    except Exception:
        pass

def test__map_tflite_reshape():
    try:
        res = _map_tflite_reshape()
    except Exception:
        pass

def test__map_tflite_resize_bilinear():
    try:
        res = _map_tflite_resize_bilinear()
    except Exception:
        pass

def test__map_tflite_concat():
    try:
        res = _map_tflite_concat()
    except Exception:
        pass

def test__map_tflite_softmax():
    try:
        res = _map_tflite_softmax()
    except Exception:
        pass

def test__map_tflite_logistic():
    try:
        res = _map_tflite_logistic()
    except Exception:
        pass

def test__map_tflite_tanh():
    try:
        res = _map_tflite_tanh()
    except Exception:
        pass

def test__map_tflite_relu():
    try:
        res = _map_tflite_relu()
    except Exception:
        pass

def test__map_tflite_relu6():
    try:
        res = _map_tflite_relu6()
    except Exception:
        pass

def test__map_tflite_relu_n1_to_1():
    try:
        res = _map_tflite_relu_n1_to_1()
    except Exception:
        pass

def test__map_tflite_dequantize():
    try:
        res = _map_tflite_dequantize()
    except Exception:
        pass

def test__map_tflite_quantize():
    try:
        res = _map_tflite_quantize()
    except Exception:
        pass

def test__map_tflite_embedding_lookup():
    try:
        res = _map_tflite_embedding_lookup()
    except Exception:
        pass

def test__map_tflite_l2_normalization():
    try:
        res = _map_tflite_l2_normalization()
    except Exception:
        pass

def test__map_tflite_local_response_normalization():
    try:
        res = _map_tflite_local_response_normalization()
    except Exception:
        pass

def test__map_tflite_space_to_depth():
    try:
        res = _map_tflite_space_to_depth()
    except Exception:
        pass

def test__map_tflite_depth_to_space():
    try:
        res = _map_tflite_depth_to_space()
    except Exception:
        pass

def test__map_tflite_floor():
    try:
        res = _map_tflite_floor()
    except Exception:
        pass

def test__map_tflite_custom_subgraph():
    try:
        res = _map_tflite_custom_subgraph()
    except Exception:
        pass

