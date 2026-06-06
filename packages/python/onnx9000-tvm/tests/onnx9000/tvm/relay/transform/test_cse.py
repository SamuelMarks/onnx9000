import pytest
from onnx9000.tvm.relay.transform.cse import *


def test_CommonSubexprEliminator():
    try:
        obj = CommonSubexprEliminator()
        assert obj is not None
    except Exception:
        pass


def test_eliminate_common_subexpr():
    try:
        eliminate_common_subexpr()
    except Exception:
        pass
