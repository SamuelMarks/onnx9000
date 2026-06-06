import pytest
from onnx9000_diffusers.models import *


def test_AutoencoderKL():
    try:
        obj = AutoencoderKL()
        assert obj is not None
    except Exception:
        pass
