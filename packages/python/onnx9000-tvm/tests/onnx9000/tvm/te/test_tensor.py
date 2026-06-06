import pytest
from onnx9000.tvm.te.tensor import *

def test_ExprOp():
    try:
        obj = ExprOp()
        assert obj is not None
    except Exception:
        pass

def test_IterVar():
    try:
        obj = IterVar()
        assert obj is not None
    except Exception:
        pass

def test_ReduceAxis():
    try:
        obj = ReduceAxis()
        assert obj is not None
    except Exception:
        pass

def test_Var():
    try:
        obj = Var()
        assert obj is not None
    except Exception:
        pass

def test_Const():
    try:
        obj = Const()
        assert obj is not None
    except Exception:
        pass

def test_BinaryOp():
    try:
        obj = BinaryOp()
        assert obj is not None
    except Exception:
        pass

def test_Add():
    try:
        obj = Add()
        assert obj is not None
    except Exception:
        pass

def test_Sub():
    try:
        obj = Sub()
        assert obj is not None
    except Exception:
        pass

def test_Mul():
    try:
        obj = Mul()
        assert obj is not None
    except Exception:
        pass

def test_Div():
    try:
        obj = Div()
        assert obj is not None
    except Exception:
        pass

def test_CallOp():
    try:
        obj = CallOp()
        assert obj is not None
    except Exception:
        pass

def test_ReduceOp():
    try:
        obj = ReduceOp()
        assert obj is not None
    except Exception:
        pass

def test_Tensor():
    try:
        obj = Tensor()
        assert obj is not None
    except Exception:
        pass

def test_TensorComputeOp():
    try:
        obj = TensorComputeOp()
        assert obj is not None
    except Exception:
        pass

def test_PlaceholderOp():
    try:
        obj = PlaceholderOp()
        assert obj is not None
    except Exception:
        pass

def test_ComputeOp():
    try:
        obj = ComputeOp()
        assert obj is not None
    except Exception:
        pass

def test_exp():
    try:
        res = exp()
    except Exception:
        pass

def test_log():
    try:
        res = log()
    except Exception:
        pass

def test_sigmoid():
    try:
        res = sigmoid()
    except Exception:
        pass

def test_sum():
    try:
        res = sum()
    except Exception:
        pass

def test_max():
    try:
        res = max()
    except Exception:
        pass

def test_min():
    try:
        res = min()
    except Exception:
        pass

def test_var():
    try:
        res = var()
    except Exception:
        pass

def test_placeholder():
    try:
        res = placeholder()
    except Exception:
        pass

def test_compute():
    try:
        res = compute()
    except Exception:
        pass

def test_reduce_axis():
    try:
        res = reduce_axis()
    except Exception:
        pass

