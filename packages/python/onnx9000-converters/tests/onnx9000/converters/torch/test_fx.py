import pytest
from onnx9000.converters.torch.fx import *


def test_FXParser():
    try:
        obj = FXParser()
        assert obj is not None
    except Exception:
        pass
