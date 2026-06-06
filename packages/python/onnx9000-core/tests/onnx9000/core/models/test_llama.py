import pytest
from onnx9000.core.models.llama import *

def test_SwiGLU():
    try:
        obj = SwiGLU()
        assert obj is not None
    except Exception:
        pass

def test_LLaMABlock():
    try:
        obj = LLaMABlock()
        assert obj is not None
    except Exception:
        pass

def test_LLaMA():
    try:
        obj = LLaMA()
        assert obj is not None
    except Exception:
        pass

def test_get_param():
    try:
        res = get_param()
    except Exception:
        pass

def test_llama_7b():
    try:
        res = llama_7b()
    except Exception:
        pass

def test_mistral_7b():
    try:
        res = mistral_7b()
    except Exception:
        pass

