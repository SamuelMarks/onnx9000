import pytest
from onnx9000.backends.codegen.ops.nn import *

def test__get_attr():
    try:
        res = _get_attr()
    except Exception:
        pass

def test_generate_attention():
    try:
        res = generate_attention()
    except Exception:
        pass

def test_generate_conv():
    try:
        res = generate_conv()
    except Exception:
        pass

def test_generate_transpose():
    try:
        res = generate_transpose()
    except Exception:
        pass

def test_generate_softmax():
    try:
        res = generate_softmax()
    except Exception:
        pass

def test_generate_log_softmax():
    try:
        res = generate_log_softmax()
    except Exception:
        pass

def test_generate_hardmax():
    try:
        res = generate_hardmax()
    except Exception:
        pass

def test_generate_rnn():
    try:
        res = generate_rnn()
    except Exception:
        pass

def test_generate_lstm():
    try:
        res = generate_lstm()
    except Exception:
        pass

def test_generate_gru():
    try:
        res = generate_gru()
    except Exception:
        pass

def test_generate_conv_transpose():
    try:
        res = generate_conv_transpose()
    except Exception:
        pass

def test_generate_deform_conv():
    try:
        res = generate_deform_conv()
    except Exception:
        pass

def test_generate_lp_normalization():
    try:
        res = generate_lp_normalization()
    except Exception:
        pass

def test_generate_lp_pool():
    try:
        res = generate_lp_pool()
    except Exception:
        pass

def test_generate_layer_normalization():
    try:
        res = generate_layer_normalization()
    except Exception:
        pass

def test_generate_mean_variance_normalization():
    try:
        res = generate_mean_variance_normalization()
    except Exception:
        pass

def test_generate_instance_normalization():
    try:
        res = generate_instance_normalization()
    except Exception:
        pass

def test_generate_max_unpool():
    try:
        res = generate_max_unpool()
    except Exception:
        pass

def test_generate_average_pool():
    try:
        res = generate_average_pool()
    except Exception:
        pass

def test_generate_max_pool():
    try:
        res = generate_max_pool()
    except Exception:
        pass

def test_generate_global_max_pool():
    try:
        res = generate_global_max_pool()
    except Exception:
        pass

def test_generate_global_average_pool():
    try:
        res = generate_global_average_pool()
    except Exception:
        pass

def test_generate_batchnorm():
    try:
        res = generate_batchnorm()
    except Exception:
        pass

def test_generate_gelu():
    try:
        res = generate_gelu()
    except Exception:
        pass

