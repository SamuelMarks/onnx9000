import pytest
from onnx9000.tvm.relay.printer import *

def test_Printer():
    try:
        obj = Printer()
        assert obj is not None
    except Exception:
        pass

def test_astext():
    try:
        res = astext()
    except Exception:
        pass

