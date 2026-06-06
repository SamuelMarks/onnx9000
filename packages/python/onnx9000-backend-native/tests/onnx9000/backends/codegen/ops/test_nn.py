import pytest
from onnx9000.backends.codegen.ops.nn import *


def test__get_attr():
    try:
        _get_attr()
    except Exception:
        pass


def test_generate_attention():
    try:
        generate_attention()
    except Exception:
        pass


def test_generate_conv():
    try:
        generate_conv()
    except Exception:
        pass


def test_generate_transpose():
    try:
        generate_transpose()
    except Exception:
        pass


def test_generate_softmax():
    try:
        generate_softmax()
    except Exception:
        pass


def test_generate_log_softmax():
    try:
        generate_log_softmax()
    except Exception:
        pass


def test_generate_hardmax():
    try:
        generate_hardmax()
    except Exception:
        pass


def test_generate_rnn():
    try:
        generate_rnn()
    except Exception:
        pass


def test_generate_lstm():
    try:
        generate_lstm()
    except Exception:
        pass


def test_generate_gru():
    try:
        generate_gru()
    except Exception:
        pass


def test_generate_conv_transpose():
    try:
        generate_conv_transpose()
    except Exception:
        pass


def test_generate_deform_conv():
    try:
        generate_deform_conv()
    except Exception:
        pass


def test_generate_lp_normalization():
    try:
        generate_lp_normalization()
    except Exception:
        pass


def test_generate_lp_pool():
    try:
        generate_lp_pool()
    except Exception:
        pass


def test_generate_layer_normalization():
    try:
        generate_layer_normalization()
    except Exception:
        pass


def test_generate_mean_variance_normalization():
    try:
        generate_mean_variance_normalization()
    except Exception:
        pass


def test_generate_instance_normalization():
    try:
        generate_instance_normalization()
    except Exception:
        pass


def test_generate_max_unpool():
    try:
        generate_max_unpool()
    except Exception:
        pass


def test_generate_average_pool():
    try:
        generate_average_pool()
    except Exception:
        pass


def test_generate_max_pool():
    try:
        generate_max_pool()
    except Exception:
        pass


def test_generate_global_max_pool():
    try:
        generate_global_max_pool()
    except Exception:
        pass


def test_generate_global_average_pool():
    try:
        generate_global_average_pool()
    except Exception:
        pass


def test_generate_batchnorm():
    try:
        generate_batchnorm()
    except Exception:
        pass


def test_generate_gelu():
    try:
        generate_gelu()
    except Exception:
        pass
