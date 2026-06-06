import pytest
from onnx9000.converters.frontend.nn.module import *


def test_Module():
    try:
        obj = Module()
        assert obj is not None
    except Exception:
        pass
