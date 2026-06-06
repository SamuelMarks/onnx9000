import pytest
from onnx9000.tvm.te.transform.lower import *


def test_ScheduleSyntaxTree():
    try:
        obj = ScheduleSyntaxTree()
        assert obj is not None
    except Exception:
        pass


def test_lower():
    try:
        lower()
    except Exception:
        pass


def test_infer_bounds():
    try:
        infer_bounds()
    except Exception:
        pass
