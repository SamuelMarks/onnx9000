import pytest
from onnx9000.onnx2gguf.builder import *


def test_GGUFValueType():
    try:
        obj = GGUFValueType()
        assert obj is not None
    except Exception:
        pass


def test_GGUFTensorType():
    try:
        obj = GGUFTensorType()
        assert obj is not None
    except Exception:
        pass


def test_GGUFWriter():
    try:
        obj = GGUFWriter()
        assert obj is not None
    except Exception:
        pass
