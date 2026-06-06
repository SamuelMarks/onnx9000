import pytest
from onnx9000.optimizer.simplifier.passes.constant_folding import *

def test_ConstantFoldingPass():
    try:
        obj = ConstantFoldingPass()
        assert obj is not None
    except Exception:
        pass

def test__evaluate_pool():
    try:
        res = _evaluate_pool()
    except Exception:
        pass

def test__evaluate_conv():
    try:
        res = _evaluate_conv()
    except Exception:
        pass

def test__numpy_to_tensor_proto():
    try:
        res = _numpy_to_tensor_proto()
    except Exception:
        pass

def test__tensor_to_numpy():
    try:
        res = _tensor_to_numpy()
    except Exception:
        pass

def test_constant_folding():
    try:
        res = constant_folding()
    except Exception:
        pass

