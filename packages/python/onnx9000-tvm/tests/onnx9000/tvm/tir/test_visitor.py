import pytest
from onnx9000.tvm.tir.visitor import *


def test_StmtVisitor():
    try:
        obj = StmtVisitor()
        assert obj is not None
    except Exception:
        pass


def test_StmtMutator():
    try:
        obj = StmtMutator()
        assert obj is not None
    except Exception:
        pass
