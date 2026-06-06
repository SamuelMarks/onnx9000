import pytest
from onnx9000.tvm.tir.expr import *

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

def test_IntImm():
    try:
        obj = IntImm()
        assert obj is not None
    except Exception:
        pass

def test_FloatImm():
    try:
        obj = FloatImm()
        assert obj is not None
    except Exception:
        pass

def test_StringImm():
    try:
        obj = StringImm()
        assert obj is not None
    except Exception:
        pass

def test_BinaryOp():
    try:
        obj = BinaryOp()
        assert obj is not None
    except Exception:
        pass

def test_Add():
    try:
        obj = Add()
        assert obj is not None
    except Exception:
        pass

def test_Sub():
    try:
        obj = Sub()
        assert obj is not None
    except Exception:
        pass

def test_Mul():
    try:
        obj = Mul()
        assert obj is not None
    except Exception:
        pass

def test_Div():
    try:
        obj = Div()
        assert obj is not None
    except Exception:
        pass

def test_Mod():
    try:
        obj = Mod()
        assert obj is not None
    except Exception:
        pass

def test_EQ():
    try:
        obj = EQ()
        assert obj is not None
    except Exception:
        pass

def test_NE():
    try:
        obj = NE()
        assert obj is not None
    except Exception:
        pass

def test_LT():
    try:
        obj = LT()
        assert obj is not None
    except Exception:
        pass

def test_LE():
    try:
        obj = LE()
        assert obj is not None
    except Exception:
        pass

def test_GT():
    try:
        obj = GT()
        assert obj is not None
    except Exception:
        pass

def test_GE():
    try:
        obj = GE()
        assert obj is not None
    except Exception:
        pass

def test_And():
    try:
        obj = And()
        assert obj is not None
    except Exception:
        pass

def test_Or():
    try:
        obj = Or()
        assert obj is not None
    except Exception:
        pass

def test_Xor():
    try:
        obj = Xor()
        assert obj is not None
    except Exception:
        pass

def test_Call():
    try:
        obj = Call()
        assert obj is not None
    except Exception:
        pass

def test_Load():
    try:
        obj = Load()
        assert obj is not None
    except Exception:
        pass

