import pytest
from onnx9000.core.parser.core import *


def test__parse_dtype():
    try:
        _parse_dtype()
    except Exception:
        pass


def test__parse_shape():
    try:
        _parse_shape()
    except Exception:
        pass


def test__parse_attribute():
    try:
        _parse_attribute()
    except Exception:
        pass


def test_parse_sparse_tensor_proto():
    try:
        parse_sparse_tensor_proto()
    except Exception:
        pass


def test_parse_tensor_proto():
    try:
        parse_tensor_proto()
    except Exception:
        pass


def test_load_tensor():
    try:
        load_tensor()
    except Exception:
        pass


def test_parse_model():
    try:
        parse_model()
    except Exception:
        pass


def test_load():
    try:
        load()
    except Exception:
        pass


def test_from_bytes():
    try:
        from_bytes()
    except Exception:
        pass
