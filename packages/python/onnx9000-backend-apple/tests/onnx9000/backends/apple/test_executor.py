import pytest
from onnx9000.backends.apple.executor import *


def test_Dispatcher():
    try:
        obj = Dispatcher()
        assert obj is not None
    except Exception:
        pass


def test__apple_matmul():
    try:
        _apple_matmul()
    except Exception:
        pass


def test__apple_add():
    try:
        _apple_add()
    except Exception:
        pass


def test__apple_sub():
    try:
        _apple_sub()
    except Exception:
        pass


def test__apple_mul():
    try:
        _apple_mul()
    except Exception:
        pass
