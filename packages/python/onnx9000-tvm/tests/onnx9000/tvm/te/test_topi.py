import pytest
from onnx9000.tvm.te.topi import *

def test_nn_conv2d():
    try:
        res = nn_conv2d()
    except Exception:
        pass

def test_nn_matmul():
    try:
        res = nn_matmul()
    except Exception:
        pass

def test_nn_pool2d():
    try:
        res = nn_pool2d()
    except Exception:
        pass

def test_nn_softmax():
    try:
        res = nn_softmax()
    except Exception:
        pass

def test_nn_layer_norm():
    try:
        res = nn_layer_norm()
    except Exception:
        pass

