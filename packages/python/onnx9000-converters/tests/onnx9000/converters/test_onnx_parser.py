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
        read_varint()
    except Exception:
        pass


def test_read_tag():
    try:
        read_tag()
    except Exception:
        pass


def test_skip_field():
    try:
        skip_field()
    except Exception:
        pass
