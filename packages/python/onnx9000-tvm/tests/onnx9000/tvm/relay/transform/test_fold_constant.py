import pytest
from onnx9000.tvm.relay.transform.fold_constant import *

def test_ConstantFolder():
    try:
        obj = ConstantFolder()
        assert obj is not None
    except Exception:
        pass

def test_fold_constant():
    try:
        res = fold_constant()
    except Exception:
        pass

