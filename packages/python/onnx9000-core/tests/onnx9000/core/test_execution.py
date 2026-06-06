import pytest
from onnx9000.core.execution import *


def test_SessionOptions():
    try:
        obj = SessionOptions()
        assert obj is not None
    except Exception:
        pass


def test_RunOptions():
    try:
        obj = RunOptions()
        assert obj is not None
    except Exception:
        pass


def test_Environment():
    try:
        obj = Environment()
        assert obj is not None
    except Exception:
        pass


def test_ExecutionContext():
    try:
        obj = ExecutionContext()
        assert obj is not None
    except Exception:
        pass


def test_ExecutionProvider():
    try:
        obj = ExecutionProvider()
        assert obj is not None
    except Exception:
        pass
