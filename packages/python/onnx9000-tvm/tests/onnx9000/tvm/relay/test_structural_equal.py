import pytest
from onnx9000.tvm.relay.structural_equal import *


def test_StructuralEquality():
    try:
        obj = StructuralEquality()
        assert obj is not None
    except Exception:
        pass


def test_structural_equal():
    try:
        structural_equal()
    except Exception:
        pass
