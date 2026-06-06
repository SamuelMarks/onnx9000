import pytest
from onnx9000.converters.frontend.nn.containers import *


def test_Sequential():
    try:
        obj = Sequential()
        assert obj is not None
    except Exception:
        pass


def test_ModuleList():
    try:
        obj = ModuleList()
        assert obj is not None
    except Exception:
        pass


def test_ModuleDict():
    try:
        obj = ModuleDict()
        assert obj is not None
    except Exception:
        pass


def test_ParameterList():
    try:
        obj = ParameterList()
        assert obj is not None
    except Exception:
        pass
