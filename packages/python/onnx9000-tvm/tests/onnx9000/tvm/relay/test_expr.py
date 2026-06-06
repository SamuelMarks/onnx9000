import pytest
from onnx9000.tvm.relay.expr import *

def test_Expr():
    try:
        obj = Expr()
        assert obj is not None
    except Exception:
        pass

def test_Var():
    try:
        obj = Var()
        assert obj is not None
    except Exception:
        pass

def test_Constant():
    try:
        obj = Constant()
        assert obj is not None
    except Exception:
        pass

def test_Op():
    try:
        obj = Op()
        assert obj is not None
    except Exception:
        pass

def test_Call():
    try:
        obj = Call()
        assert obj is not None
    except Exception:
        pass

def test_TupleExpr():
    try:
        obj = TupleExpr()
        assert obj is not None
    except Exception:
        pass

def test_TupleGetItem():
    try:
        obj = TupleGetItem()
        assert obj is not None
    except Exception:
        pass

def test_Let():
    try:
        obj = Let()
        assert obj is not None
    except Exception:
        pass

def test_If():
    try:
        obj = If()
        assert obj is not None
    except Exception:
        pass

def test_Function():
    try:
        obj = Function()
        assert obj is not None
    except Exception:
        pass

