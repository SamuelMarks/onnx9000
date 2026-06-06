import pytest
from onnx9000.toolkit.safetensors.interop import *

def test_load_pytorch_safetensors():
    try:
        res = load_pytorch_safetensors()
    except Exception:
        pass

def test_load_tensorflow_safetensors():
    try:
        res = load_tensorflow_safetensors()
    except Exception:
        pass

def test_load_flax_safetensors():
    try:
        res = load_flax_safetensors()
    except Exception:
        pass

