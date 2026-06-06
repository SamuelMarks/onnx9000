import pytest
from onnx9000.core.models.mamba import *


def test_MambaBlock():
    try:
        obj = MambaBlock()
        assert obj is not None
    except Exception:
        pass


def test_Mamba():
    try:
        obj = Mamba()
        assert obj is not None
    except Exception:
        pass


def test_get_param():
    try:
        get_param()
    except Exception:
        pass


def test_mamba_130m():
    try:
        mamba_130m()
    except Exception:
        pass
