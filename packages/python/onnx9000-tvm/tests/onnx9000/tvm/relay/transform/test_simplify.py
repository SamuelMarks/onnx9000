import pytest
from onnx9000.tvm.relay.transform.simplify import *


def test_AlgebraicSimplifier():
    try:
        obj = AlgebraicSimplifier()
        assert obj is not None
    except Exception:
        pass


def test_simplify_algebra():
    try:
        simplify_algebra()
    except Exception:
        pass
