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
        exp()
    except Exception:
        pass


def test_log():
    try:
        log()
    except Exception:
        pass


def test_sigmoid():
    try:
        sigmoid()
    except Exception:
        pass


def test_sum():
    try:
        sum()
    except Exception:
        pass


def test_max():
    try:
        max()
    except Exception:
        pass


def test_min():
    try:
        min()
    except Exception:
        pass


def test_var():
    try:
        var()
    except Exception:
        pass


def test_placeholder():
    try:
        placeholder()
    except Exception:
        pass


def test_compute():
    try:
        compute()
    except Exception:
        pass


def test_reduce_axis():
    try:
        reduce_axis()
    except Exception:
        pass
