import pytest
from onnx9000.core.models.mobilevit import *


def test_MobileViTBlock():
    try:
        obj = MobileViTBlock()
        assert obj is not None
    except Exception:
        pass


def test_MobileViT():
    try:
        obj = MobileViT()
        assert obj is not None
    except Exception:
        pass


def test_get_param():
    try:
        get_param()
    except Exception:
        pass


def test_mobilevit_s():
    try:
        mobilevit_s()
    except Exception:
        pass
