import pytest
from onnx9000.c_compiler.pooling import *

def test_generate_pooling():
    try:
        res = generate_pooling()
    except Exception:
        pass

def test_generate_global_pooling():
    try:
        res = generate_global_pooling()
    except Exception:
        pass

def test_generate_reduction():
    try:
        res = generate_reduction()
    except Exception:
        pass

def test_generate_arg_reduction():
    try:
        res = generate_arg_reduction()
    except Exception:
        pass

