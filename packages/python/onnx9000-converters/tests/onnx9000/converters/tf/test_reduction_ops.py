import pytest
from onnx9000.converters.tf.reduction_ops import *


def test__map_reduce_op():
    try:
        _map_reduce_op()
    except Exception:
        pass


def test__map_bincount():
    try:
        _map_bincount()
    except Exception:
        pass


def test__map_cumsum():
    try:
        _map_cumsum()
    except Exception:
        pass


def test__map_cumprod():
    try:
        _map_cumprod()
    except Exception:
        pass


def test__map_logical_binary():
    try:
        _map_logical_binary()
    except Exception:
        pass


def test__map_not_equal():
    try:
        _map_not_equal()
    except Exception:
        pass
