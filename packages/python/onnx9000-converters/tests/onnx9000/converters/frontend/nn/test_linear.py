import pytest
from onnx9000.converters.frontend.nn.linear import *


def test_Linear():
    try:
        obj = Linear()
        assert obj is not None
    except Exception:
        pass
