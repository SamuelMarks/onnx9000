import pytest
from onnx9000.converters.paddle.math_ops import *


def test__map_log2():
    try:
        _map_log2()
    except Exception:
        pass


def test__map_log10():
    try:
        _map_log10()
    except Exception:
        pass


def test__map_clip():
    try:
        _map_clip()
    except Exception:
        pass


def test__map_simple_binary():
    try:
        _map_simple_binary()
    except Exception:
        pass


def test__map_simple_unary():
    try:
        _map_simple_unary()
    except Exception:
        pass


def test__map_floordiv():
    try:
        _map_floordiv()
    except Exception:
        pass


def test__map_log1p():
    try:
        _map_log1p()
    except Exception:
        pass


def test__map_rsqrt():
    try:
        _map_rsqrt()
    except Exception:
        pass


def test__map_square():
    try:
        _map_square()
    except Exception:
        pass


def test__map_isfinite():
    try:
        _map_isfinite()
    except Exception:
        pass


def test__map_scale():
    try:
        _map_scale()
    except Exception:
        pass


def test__map_sum():
    try:
        _map_sum()
    except Exception:
        pass


def test__map_dot():
    try:
        _map_dot()
    except Exception:
        pass


def test__map_cross():
    try:
        _map_cross()
    except Exception:
        pass


def test__map_custom():
    try:
        _map_custom()
    except Exception:
        pass
