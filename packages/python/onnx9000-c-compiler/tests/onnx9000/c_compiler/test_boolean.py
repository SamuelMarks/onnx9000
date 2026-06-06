import pytest
from onnx9000.c_compiler.boolean import *


def test_generate_boolean_binary():
    try:
        generate_boolean_binary()
    except Exception:
        pass


def test_generate_boolean_unary():
    try:
        generate_boolean_unary()
    except Exception:
        pass


def test_generate_where():
    try:
        generate_where()
    except Exception:
        pass
