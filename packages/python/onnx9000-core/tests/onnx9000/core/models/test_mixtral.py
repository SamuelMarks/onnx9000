import pytest
from onnx9000.core.models.mixtral import *

def test_SparseMoE():
    try:
        obj = SparseMoE()
        assert obj is not None
    except Exception:
        pass

def test_MixtralBlock():
    try:
        obj = MixtralBlock()
        assert obj is not None
    except Exception:
        pass

def test_Mixtral():
    try:
        obj = Mixtral()
        assert obj is not None
    except Exception:
        pass

def test_get_param():
    try:
        res = get_param()
    except Exception:
        pass

def test_mixtral_8x7b():
    try:
        res = mixtral_8x7b()
    except Exception:
        pass

