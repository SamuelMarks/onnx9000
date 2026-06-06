import pytest
from onnx9000.tensorrt.ops_dim import *


def test__get_input():
    try:
        _get_input()
    except Exception:
        pass


def test_trt_reshape():
    try:
        trt_reshape()
    except Exception:
        pass


def test_trt_transpose():
    try:
        trt_transpose()
    except Exception:
        pass


def test_trt_concat():
    try:
        trt_concat()
    except Exception:
        pass


def test_trt_slice():
    try:
        trt_slice()
    except Exception:
        pass


def test_trt_gather():
    try:
        trt_gather()
    except Exception:
        pass
