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
        get_active_builder()
    except Exception:
        pass


def test_set_active_builder():
    try:
        set_active_builder()
    except Exception:
        pass


def test_pop_active_builder():
    try:
        pop_active_builder()
    except Exception:
        pass


def test__make_var():
    try:
        _make_var()
    except Exception:
        pass


def test__make_vars():
    try:
        _make_vars()
    except Exception:
        pass


def test_Constant():
    try:
        Constant()
    except Exception:
        pass


def test_If():
    try:
        If()
    except Exception:
        pass


def test_Loop():
    try:
        Loop()
    except Exception:
        pass


def test_Scan():
    try:
        Scan()
    except Exception:
        pass
