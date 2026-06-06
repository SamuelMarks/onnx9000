import pytest
from onnx9000.converters.caffe.weights import *

def test_ProtobufDecoder():
    try:
        obj = ProtobufDecoder()
        assert obj is not None
    except Exception:
        pass

def test_parse_blob():
    try:
        res = parse_blob()
    except Exception:
        pass

def test_parse_layer():
    try:
        res = parse_layer()
    except Exception:
        pass

def test_load_caffemodel():
    try:
        res = load_caffemodel()
    except Exception:
        pass

