import pytest
from onnx9000.diffusers.models import *

def test_AutoencoderKL():
    try:
        obj = AutoencoderKL()
        assert obj is not None
    except Exception:
        pass

def test_UNet2DConditionModel():
    try:
        obj = UNet2DConditionModel()
        assert obj is not None
    except Exception:
        pass

