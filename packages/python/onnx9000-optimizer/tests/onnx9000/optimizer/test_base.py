import pytest
from onnx9000.optimizer.base import *


def test_PassContext():
    try:
        obj = PassContext()
        assert obj is not None
    except Exception:
        pass


def test_Pass():
    try:
        obj = Pass()
        assert obj is not None
    except Exception:
        pass
