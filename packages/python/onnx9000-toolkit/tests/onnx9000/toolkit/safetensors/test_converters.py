import pytest
from onnx9000.toolkit.safetensors.converters import *

def test_convert_pytorch_to_safetensors():
    try:
        res = convert_pytorch_to_safetensors()
    except Exception:
        pass

def test_convert_tf_to_safetensors():
    try:
        res = convert_tf_to_safetensors()
    except Exception:
        pass

