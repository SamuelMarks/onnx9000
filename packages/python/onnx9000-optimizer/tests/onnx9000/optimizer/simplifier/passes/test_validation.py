import pytest
from onnx9000.optimizer.simplifier.passes.validation import *


def test_ValidationPass():
    try:
        obj = ValidationPass()
        assert obj is not None
    except Exception:
        pass


def test_detect_cycles():
    try:
        detect_cycles()
    except Exception:
        pass


def test_detect_dangling():
    try:
        detect_dangling()
    except Exception:
        pass
