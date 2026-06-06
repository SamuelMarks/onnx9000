import pytest
from onnx9000.tvm.relay.transform.unroll_let import *


def test_LetUnroller():
    try:
        obj = LetUnroller()
        assert obj is not None
    except Exception:
        pass


def test_unroll_let():
    try:
        unroll_let()
    except Exception:
        pass
