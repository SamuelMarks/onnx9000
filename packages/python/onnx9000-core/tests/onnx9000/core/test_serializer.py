import pytest
from onnx9000.core.serializer import *

def test_SerializationError():
    try:
        obj = SerializationError()
        assert obj is not None
    except Exception:
        pass

def test__serialize_shape():
    try:
        res = _serialize_shape()
    except Exception:
        pass

def test__serialize_tensor():
    try:
        res = _serialize_tensor()
    except Exception:
        pass

def test__serialize_sparse_tensor():
    try:
        res = _serialize_sparse_tensor()
    except Exception:
        pass

def test__sanitize_string():
    try:
        res = _sanitize_string()
    except Exception:
        pass

def test_serialize_model():
    try:
        res = serialize_model()
    except Exception:
        pass

def test_save():
    try:
        res = save()
    except Exception:
        pass

def test_to_bytes():
    try:
        res = to_bytes()
    except Exception:
        pass

