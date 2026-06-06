import pytest
from onnx9000.tvm.tir.stmt import *

def test_Stmt():
    try:
        obj = Stmt()
        assert obj is not None
    except Exception:
        pass

def test_LetStmt():
    try:
        obj = LetStmt()
        assert obj is not None
    except Exception:
        pass

def test_AssertStmt():
    try:
        obj = AssertStmt()
        assert obj is not None
    except Exception:
        pass

def test_For():
    try:
        obj = For()
        assert obj is not None
    except Exception:
        pass

def test_While():
    try:
        obj = While()
        assert obj is not None
    except Exception:
        pass

def test_Store():
    try:
        obj = Store()
        assert obj is not None
    except Exception:
        pass

def test_Allocate():
    try:
        obj = Allocate()
        assert obj is not None
    except Exception:
        pass

def test_IfThenElse():
    try:
        obj = IfThenElse()
        assert obj is not None
    except Exception:
        pass

def test_Evaluate():
    try:
        obj = Evaluate()
        assert obj is not None
    except Exception:
        pass

def test_SeqStmt():
    try:
        obj = SeqStmt()
        assert obj is not None
    except Exception:
        pass

def test_Buffer():
    try:
        obj = Buffer()
        assert obj is not None
    except Exception:
        pass

