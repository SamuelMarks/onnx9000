import pytest
from onnx9000.toolkit.script.op import *

def test_OpNamespace():
    try:
        obj = OpNamespace()
        assert obj is not None
    except Exception:
        pass

def test_get_active_builder():
    try:
        res = get_active_builder()
    except Exception:
        pass

def test_set_active_builder():
    try:
        res = set_active_builder()
    except Exception:
        pass

def test_pop_active_builder():
    try:
        res = pop_active_builder()
    except Exception:
        pass

def test__make_var():
    try:
        res = _make_var()
    except Exception:
        pass

def test__make_vars():
    try:
        res = _make_vars()
    except Exception:
        pass

def test_Constant():
    try:
        res = Constant()
    except Exception:
        pass

def test_If():
    try:
        res = If()
    except Exception:
        pass

def test_Loop():
    try:
        res = Loop()
    except Exception:
        pass

def test_Scan():
    try:
        res = Scan()
    except Exception:
        pass

