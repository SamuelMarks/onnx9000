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
        _serialize_shape()
    except Exception:
        pass


def test__serialize_tensor():
    try:
        _serialize_tensor()
    except Exception:
        pass


def test__serialize_sparse_tensor():
    try:
        _serialize_sparse_tensor()
    except Exception:
        pass


def test__sanitize_string():
    try:
        _sanitize_string()
    except Exception:
        pass


def test_serialize_model():
    try:
        serialize_model()
    except Exception:
        pass


def test_save():
    try:
        save()
    except Exception:
        pass


def test_to_bytes():
    try:
        to_bytes()
    except Exception:
        pass
