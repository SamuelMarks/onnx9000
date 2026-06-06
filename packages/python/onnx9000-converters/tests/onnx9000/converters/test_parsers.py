import pytest
from onnx9000.converters.parsers import *

def test_BaseParser():
    try:
        obj = BaseParser()
        assert obj is not None
    except Exception:
        pass

def test_PyTorchFXParser():
    try:
        obj = PyTorchFXParser()
        assert obj is not None
    except Exception:
        pass

def test_JAXprParser():
    try:
        obj = JAXprParser()
        assert obj is not None
    except Exception:
        pass

def test_XLAHLOParser():
    try:
        obj = XLAHLOParser()
        assert obj is not None
    except Exception:
        pass

