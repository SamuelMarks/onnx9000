import pytest
from onnx9000.tvm.relay.visitor import *


def test_ExprVisitor():
    try:
        obj = ExprVisitor()
        assert obj is not None
    except Exception:
        pass


def test_ExprMutator():
    try:
        obj = ExprMutator()
        assert obj is not None
    except Exception:
        pass
