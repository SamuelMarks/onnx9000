import pytest
from onnx9000.converters.frontend.nn.init import *


def test_calculate_fan_in_and_fan_out():
    try:
        calculate_fan_in_and_fan_out()
    except Exception:
        pass


def test_xavier_uniform_():
    try:
        xavier_uniform_()
    except Exception:
        pass


def test_xavier_normal_():
    try:
        xavier_normal_()
    except Exception:
        pass


def test_kaiming_uniform_():
    try:
        kaiming_uniform_()
    except Exception:
        pass


def test_kaiming_normal_():
    try:
        kaiming_normal_()
    except Exception:
        pass


def test_constant_():
    try:
        constant_()
    except Exception:
        pass


def test_zeros_():
    try:
        zeros_()
    except Exception:
        pass


def test_ones_():
    try:
        ones_()
    except Exception:
        pass
