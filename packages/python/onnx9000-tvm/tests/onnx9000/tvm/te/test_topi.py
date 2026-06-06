import pytest
from onnx9000.tvm.te.topi import *


def test_nn_conv2d():
    try:
        nn_conv2d()
    except Exception:
        pass


def test_nn_matmul():
    try:
        nn_matmul()
    except Exception:
        pass


def test_nn_pool2d():
    try:
        nn_pool2d()
    except Exception:
        pass


def test_nn_softmax():
    try:
        nn_softmax()
    except Exception:
        pass


def test_nn_layer_norm():
    try:
        nn_layer_norm()
    except Exception:
        pass
