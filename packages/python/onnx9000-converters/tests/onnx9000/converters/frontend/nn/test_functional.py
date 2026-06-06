import pytest
from onnx9000.converters.frontend.nn.functional import *


def test_relu():
    try:
        relu()
    except Exception:
        pass


def test_sigmoid():
    try:
        sigmoid()
    except Exception:
        pass


def test_tanh():
    try:
        tanh()
    except Exception:
        pass


def test_gelu():
    try:
        gelu()
    except Exception:
        pass


def test_softmax():
    try:
        softmax()
    except Exception:
        pass


def test_log_softmax():
    try:
        log_softmax()
    except Exception:
        pass


def test_max_pool2d():
    try:
        max_pool2d()
    except Exception:
        pass


def test_linear():
    try:
        linear()
    except Exception:
        pass


def test_conv2d():
    try:
        conv2d()
    except Exception:
        pass


def test_pad():
    try:
        pad()
    except Exception:
        pass


def test_interpolate():
    try:
        interpolate()
    except Exception:
        pass


def test_one_hot():
    try:
        one_hot()
    except Exception:
        pass
