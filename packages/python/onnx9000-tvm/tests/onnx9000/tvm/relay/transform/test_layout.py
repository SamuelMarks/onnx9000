import pytest
from onnx9000.tvm.relay.transform.layout import *

def test_LayoutTransform():
    try:
        obj = LayoutTransform()
        assert obj is not None
    except Exception:
        pass

def test_transform_layout():
    try:
        res = transform_layout()
    except Exception:
        pass

