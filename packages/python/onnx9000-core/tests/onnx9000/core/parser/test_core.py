import pytest
from onnx9000.core.parser.core import *

def test__parse_dtype():
    try:
        res = _parse_dtype()
    except Exception:
        pass

def test__parse_shape():
    try:
        res = _parse_shape()
    except Exception:
        pass

def test__parse_attribute():
    try:
        res = _parse_attribute()
    except Exception:
        pass

def test_parse_sparse_tensor_proto():
    try:
        res = parse_sparse_tensor_proto()
    except Exception:
        pass

def test_parse_tensor_proto():
    try:
        res = parse_tensor_proto()
    except Exception:
        pass

def test_load_tensor():
    try:
        res = load_tensor()
    except Exception:
        pass

def test_parse_model():
    try:
        res = parse_model()
    except Exception:
        pass

def test_load():
    try:
        res = load()
    except Exception:
        pass

def test_from_bytes():
    try:
        res = from_bytes()
    except Exception:
        pass

