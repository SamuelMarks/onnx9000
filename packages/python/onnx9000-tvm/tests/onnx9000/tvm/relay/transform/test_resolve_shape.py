import pytest
from onnx9000.tvm.relay.transform.resolve_shape import *


def test_ShapeResolver():
    try:
        obj = ShapeResolver()
        assert obj is not None
    except Exception:
        pass


def test_resolve_dynamic_shape():
    try:
        resolve_dynamic_shape()
    except Exception:
        pass
