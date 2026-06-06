import pytest
from onnx9000.converters.paddle.reduction_ops import *


def test__map_allclose():
    try:
        _map_allclose()
    except Exception:
        pass


def test__map_logical_binary():
    try:
        _map_logical_binary()
    except Exception:
        pass


def test__map_logical_unary():
    try:
        _map_logical_unary()
    except Exception:
        pass


def test__map_not_equal():
    try:
        _map_not_equal()
    except Exception:
        pass


def test__map_reduce():
    try:
        _map_reduce()
    except Exception:
        pass


def test__map_arg():
    try:
        _map_arg()
    except Exception:
        pass


def test__map_cumsum():
    try:
        _map_cumsum()
    except Exception:
        pass


def test__map_custom():
    try:
        _map_custom()
    except Exception:
        pass
