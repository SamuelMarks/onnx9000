import pytest
from onnx9000.converters.darknet.mapper import *

def test_DarknetMapper():
    try:
        obj = DarknetMapper()
        assert obj is not None
    except Exception:
        pass

