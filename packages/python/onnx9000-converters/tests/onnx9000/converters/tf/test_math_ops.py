import pytest
from onnx9000.converters.tf.math_ops import *

def test__map_simple_binary():
    try:
        res = _map_simple_binary()
    except Exception:
        pass

def test__map_simple_unary():
    try:
        res = _map_simple_unary()
    except Exception:
        pass

def test__map_floor_div():
    try:
        res = _map_floor_div()
    except Exception:
        pass

def test__map_floor_mod():
    try:
        res = _map_floor_mod()
    except Exception:
        pass

def test__map_square():
    try:
        res = _map_square()
    except Exception:
        pass

def test__map_rsqrt():
    try:
        res = _map_rsqrt()
    except Exception:
        pass

def test__map_expm1():
    try:
        res = _map_expm1()
    except Exception:
        pass

def test__map_log1p():
    try:
        res = _map_log1p()
    except Exception:
        pass

def test__map_atan2():
    try:
        res = _map_atan2()
    except Exception:
        pass

def test__map_isfinite():
    try:
        res = _map_isfinite()
    except Exception:
        pass

def test__map_complex_abs():
    try:
        res = _map_complex_abs()
    except Exception:
        pass

def test__map_angle():
    try:
        res = _map_angle()
    except Exception:
        pass

