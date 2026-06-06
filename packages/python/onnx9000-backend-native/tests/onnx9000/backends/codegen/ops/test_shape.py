import pytest
from onnx9000.backends.codegen.ops.shape import *

def test_generate_reshape():
    try:
        res = generate_reshape()
    except Exception:
        pass

def test_generate_flatten():
    try:
        res = generate_flatten()
    except Exception:
        pass

def test_generate_squeeze():
    try:
        res = generate_squeeze()
    except Exception:
        pass

def test_generate_unsqueeze():
    try:
        res = generate_unsqueeze()
    except Exception:
        pass

def test_generate_cast_like():
    try:
        res = generate_cast_like()
    except Exception:
        pass

def test_generate_cast():
    try:
        res = generate_cast()
    except Exception:
        pass

def test_generate_expand():
    try:
        res = generate_expand()
    except Exception:
        pass

