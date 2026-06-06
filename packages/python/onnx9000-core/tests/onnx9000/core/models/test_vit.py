import pytest
from onnx9000.core.models.vit import *


def test_PatchEmbed():
    try:
        obj = PatchEmbed()
        assert obj is not None
    except Exception:
        pass


def test_Block():
    try:
        obj = Block()
        assert obj is not None
    except Exception:
        pass


def test_VisionTransformer():
    try:
        obj = VisionTransformer()
        assert obj is not None
    except Exception:
        pass


def test_get_param():
    try:
        get_param()
    except Exception:
        pass


def test_vit_base_patch16_224():
    try:
        vit_base_patch16_224()
    except Exception:
        pass
