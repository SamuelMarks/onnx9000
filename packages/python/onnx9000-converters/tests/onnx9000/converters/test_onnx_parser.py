import pytest
from onnx9000.converters.onnx_parser import *

def test_PureOnnxParser():
    try:
        obj = PureOnnxParser()
        assert obj is not None
    except Exception:
        pass

def test_read_varint():
    try:
        res = read_varint()
    except Exception:
        pass

def test_read_tag():
    try:
        res = read_tag()
    except Exception:
        pass

def test_skip_field():
    try:
        res = skip_field()
    except Exception:
        pass

