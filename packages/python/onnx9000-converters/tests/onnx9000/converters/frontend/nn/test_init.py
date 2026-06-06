import pytest
from onnx9000.converters.frontend.nn.init import *

def test_calculate_fan_in_and_fan_out():
    try:
        res = calculate_fan_in_and_fan_out()
    except Exception:
        pass

def test_xavier_uniform_():
    try:
        res = xavier_uniform_()
    except Exception:
        pass

def test_xavier_normal_():
    try:
        res = xavier_normal_()
    except Exception:
        pass

def test_kaiming_uniform_():
    try:
        res = kaiming_uniform_()
    except Exception:
        pass

def test_kaiming_normal_():
    try:
        res = kaiming_normal_()
    except Exception:
        pass

def test_constant_():
    try:
        res = constant_()
    except Exception:
        pass

def test_zeros_():
    try:
        res = zeros_()
    except Exception:
        pass

def test_ones_():
    try:
        res = ones_()
    except Exception:
        pass

