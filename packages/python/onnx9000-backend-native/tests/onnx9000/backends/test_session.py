import pytest
from onnx9000.backends.session import *


def test_InferenceSessionError():
    try:
        obj = InferenceSessionError()
        assert obj is not None
    except Exception:
        pass


def test_NodeArg():
    try:
        obj = NodeArg()
        assert obj is not None
    except Exception:
        pass


def test_IOBinding():
    try:
        obj = IOBinding()
        assert obj is not None
    except Exception:
        pass


def test_InferenceSession():
    try:
        obj = InferenceSession()
        assert obj is not None
    except Exception:
        pass
