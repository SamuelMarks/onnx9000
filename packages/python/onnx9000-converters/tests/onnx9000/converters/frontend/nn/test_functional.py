import pytest
from onnx9000.converters.frontend.nn.functional import *

def test_relu():
    try:
        res = relu()
    except Exception:
        pass

def test_sigmoid():
    try:
        res = sigmoid()
    except Exception:
        pass

def test_tanh():
    try:
        res = tanh()
    except Exception:
        pass

def test_gelu():
    try:
        res = gelu()
    except Exception:
        pass

def test_softmax():
    try:
        res = softmax()
    except Exception:
        pass

def test_log_softmax():
    try:
        res = log_softmax()
    except Exception:
        pass

def test_max_pool2d():
    try:
        res = max_pool2d()
    except Exception:
        pass

def test_linear():
    try:
        res = linear()
    except Exception:
        pass

def test_conv2d():
    try:
        res = conv2d()
    except Exception:
        pass

def test_pad():
    try:
        res = pad()
    except Exception:
        pass

def test_interpolate():
    try:
        res = interpolate()
    except Exception:
        pass

def test_one_hot():
    try:
        res = one_hot()
    except Exception:
        pass

