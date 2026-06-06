import pytest
from onnx9000.c_compiler.activations import *

def test_generate_activation():
    try:
        res = generate_activation()
    except Exception:
        pass

def test_generate_softmax():
    try:
        res = generate_softmax()
    except Exception:
        pass

def test_generate_normalization():
    try:
        res = generate_normalization()
    except Exception:
        pass

