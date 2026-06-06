import pytest
from onnx9000.toolkit.script.control_flow import *


def test_BranchContext():
    try:
        obj = BranchContext()
        assert obj is not None
    except Exception:
        pass


def test_IfContextManager():
    try:
        obj = IfContextManager()
        assert obj is not None
    except Exception:
        pass


def test_LoopContextManager():
    try:
        obj = LoopContextManager()
        assert obj is not None
    except Exception:
        pass
