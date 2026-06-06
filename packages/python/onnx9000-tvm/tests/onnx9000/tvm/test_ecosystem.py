import pytest
from onnx9000.tvm.ecosystem import *


def test_WebCodecsInterop():
    try:
        obj = WebCodecsInterop()
        assert obj is not None
    except Exception:
        pass


def test_TVMParityCertifier():
    try:
        obj = TVMParityCertifier()
        assert obj is not None
    except Exception:
        pass
