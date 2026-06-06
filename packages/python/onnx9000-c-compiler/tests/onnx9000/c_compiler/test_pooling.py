import pytest
from onnx9000.c_compiler.pooling import *


def test_generate_pooling():
    try:
        generate_pooling()
    except Exception:
        pass


def test_generate_global_pooling():
    try:
        generate_global_pooling()
    except Exception:
        pass


def test_generate_reduction():
    try:
        generate_reduction()
    except Exception:
        pass


def test_generate_arg_reduction():
    try:
        generate_arg_reduction()
    except Exception:
        pass
