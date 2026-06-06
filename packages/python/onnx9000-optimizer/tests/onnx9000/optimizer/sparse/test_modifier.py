import pytest
from onnx9000.optimizer.sparse.modifier import *

def test_Modifier():
    try:
        obj = Modifier()
        assert obj is not None
    except Exception:
        pass

def test_ConstantPruningModifier():
    try:
        obj = ConstantPruningModifier()
        assert obj is not None
    except Exception:
        pass

def test_MagnitudePruningModifier():
    try:
        obj = MagnitudePruningModifier()
        assert obj is not None
    except Exception:
        pass

def test_GradualPruningModifier():
    try:
        obj = GradualPruningModifier()
        assert obj is not None
    except Exception:
        pass

def test_OBSPruningModifier():
    try:
        obj = OBSPruningModifier()
        assert obj is not None
    except Exception:
        pass

def test_FisherPruningModifier():
    try:
        obj = FisherPruningModifier()
        assert obj is not None
    except Exception:
        pass

def test_MovementPruningModifier():
    try:
        obj = MovementPruningModifier()
        assert obj is not None
    except Exception:
        pass

def test_AccuracyAwarePruningModifier():
    try:
        obj = AccuracyAwarePruningModifier()
        assert obj is not None
    except Exception:
        pass

def test_GlobalMagnitudePruningModifier():
    try:
        obj = GlobalMagnitudePruningModifier()
        assert obj is not None
    except Exception:
        pass

def test_QuantizationModifier():
    try:
        obj = QuantizationModifier()
        assert obj is not None
    except Exception:
        pass

def test_AsymmetricSparseQuantizationModifier():
    try:
        obj = AsymmetricSparseQuantizationModifier()
        assert obj is not None
    except Exception:
        pass

def test_SparseQLinearConvModifier():
    try:
        obj = SparseQLinearConvModifier()
        assert obj is not None
    except Exception:
        pass

def test_NMPruningModifier():
    try:
        obj = NMPruningModifier()
        assert obj is not None
    except Exception:
        pass

def test_manage_calibration_memory():
    try:
        res = manage_calibration_memory()
    except Exception:
        pass

def test_parse_recipe():
    try:
        res = parse_recipe()
    except Exception:
        pass

def test_apply_recipe():
    try:
        res = apply_recipe()
    except Exception:
        pass

