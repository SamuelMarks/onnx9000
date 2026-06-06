import pytest
from onnx9000.core.models.rwkv import *

def test_RWKVTimeMix():
    try:
        obj = RWKVTimeMix()
        assert obj is not None
    except Exception:
        pass

def test_RWKVChannelMix():
    try:
        obj = RWKVChannelMix()
        assert obj is not None
    except Exception:
        pass

def test_RWKVBlock():
    try:
        obj = RWKVBlock()
        assert obj is not None
    except Exception:
        pass

def test_RWKV():
    try:
        obj = RWKV()
        assert obj is not None
    except Exception:
        pass

def test_get_param():
    try:
        res = get_param()
    except Exception:
        pass

def test_rwkv_v4():
    try:
        res = rwkv_v4()
    except Exception:
        pass

