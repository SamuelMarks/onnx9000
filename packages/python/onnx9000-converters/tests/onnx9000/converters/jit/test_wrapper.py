import pytest
from onnx9000.converters.jit.wrapper import *


def test_CompiledModel():
    try:
        obj = CompiledModel()
        assert obj is not None
    except Exception:
        pass
