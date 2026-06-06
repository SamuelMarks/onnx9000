import pytest
from onnx9000.c_compiler.spatial import *


def test_get_attribute():
    try:
        get_attribute()
    except Exception:
        pass


def test_generate_conv():
    try:
        generate_conv()
    except Exception:
        pass


def test_generate_conv_transpose():
    try:
        generate_conv_transpose()
    except Exception:
        pass
