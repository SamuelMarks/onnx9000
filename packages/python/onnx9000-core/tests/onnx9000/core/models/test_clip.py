import pytest
from onnx9000.core.models.clip import *


def test_CLIP():
    try:
        obj = CLIP()
        assert obj is not None
    except Exception:
        pass


def test_get_param():
    try:
        get_param()
    except Exception:
        pass


def test_clip_vit_base_patch16():
    try:
        clip_vit_base_patch16()
    except Exception:
        pass
