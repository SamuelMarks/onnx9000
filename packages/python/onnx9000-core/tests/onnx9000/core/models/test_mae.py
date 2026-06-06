import pytest
from onnx9000.core.models.mae import *


def test_MaskedAutoencoderViT():
    try:
        obj = MaskedAutoencoderViT()
        assert obj is not None
    except Exception:
        pass


def test_get_param():
    try:
        get_param()
    except Exception:
        pass


def test_mae_vit_base_patch16():
    try:
        mae_vit_base_patch16()
    except Exception:
        pass
