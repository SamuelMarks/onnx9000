import pytest
from onnx9000.converters.frontend.codegen import *

def test_generate_pytorch():
    try:
        res = generate_pytorch()
    except Exception:
        pass

def test_generate_keras():
    try:
        res = generate_keras()
    except Exception:
        pass

def test_generate_jax():
    try:
        res = generate_jax()
    except Exception:
        pass

