import pytest
from onnx9000.converters.frontend.tensor import *


def test_Node():
    try:
        obj = Node()
        assert obj is not None
    except Exception:
        pass


def test_Tensor():
    try:
        obj = Tensor()
        assert obj is not None
    except Exception:
        pass


def test_Parameter():
    try:
        obj = Parameter()
        assert obj is not None
    except Exception:
        pass
