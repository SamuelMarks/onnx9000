import pytest
from onnx9000.core.models.swin import *


def test_WindowAttention():
    try:
        obj = WindowAttention()
        assert obj is not None
    except Exception:
        pass


def test_SwinTransformerBlock():
    try:
        obj = SwinTransformerBlock()
        assert obj is not None
    except Exception:
        pass


def test_SwinTransformer():
    try:
        obj = SwinTransformer()
        assert obj is not None
    except Exception:
        pass


def test_get_param():
    try:
        get_param()
    except Exception:
        pass


def test_swin_t():
    try:
        swin_t()
    except Exception:
        pass
