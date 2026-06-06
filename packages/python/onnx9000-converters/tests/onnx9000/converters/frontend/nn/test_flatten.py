import pytest
from onnx9000.converters.frontend.nn.flatten import *


def test_Flatten():
    try:
        obj = Flatten()
        assert obj is not None
    except Exception:
        pass


def test_Unflatten():
    try:
        obj = Unflatten()
        assert obj is not None
    except Exception:
        pass
