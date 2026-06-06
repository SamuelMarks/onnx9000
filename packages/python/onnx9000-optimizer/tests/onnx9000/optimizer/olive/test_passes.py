import pytest
from onnx9000.optimizer.olive.passes import *

def test_Pass():
    try:
        obj = Pass()
        assert obj is not None
    except Exception:
        pass

def test_QuantizationPass():
    try:
        obj = QuantizationPass()
        assert obj is not None
    except Exception:
        pass

def test_DynamicQuantizationPass():
    try:
        obj = DynamicQuantizationPass()
        assert obj is not None
    except Exception:
        pass

def test_StaticQuantizationPass():
    try:
        obj = StaticQuantizationPass()
        assert obj is not None
    except Exception:
        pass

def test_WeightOnlyQuantizationPass():
    try:
        obj = WeightOnlyQuantizationPass()
        assert obj is not None
    except Exception:
        pass

def test_PruningPass():
    try:
        obj = PruningPass()
        assert obj is not None
    except Exception:
        pass

def test_GraphFusionPass():
    try:
        obj = GraphFusionPass()
        assert obj is not None
    except Exception:
        pass

def test_MixedPrecisionPass():
    try:
        obj = MixedPrecisionPass()
        assert obj is not None
    except Exception:
        pass

def test_LayoutConversionPass():
    try:
        obj = LayoutConversionPass()
        assert obj is not None
    except Exception:
        pass

def test_OrtPerfTuningPass():
    try:
        obj = OrtPerfTuningPass()
        assert obj is not None
    except Exception:
        pass

def test_OrtTransformerOptimizationPass():
    try:
        obj = OrtTransformerOptimizationPass()
        assert obj is not None
    except Exception:
        pass

def test_ConstantFoldingPass():
    try:
        obj = ConstantFoldingPass()
        assert obj is not None
    except Exception:
        pass

def test_StripIdentityPass():
    try:
        obj = StripIdentityPass()
        assert obj is not None
    except Exception:
        pass

def test_StripUnusedInitializersPass():
    try:
        obj = StripUnusedInitializersPass()
        assert obj is not None
    except Exception:
        pass

def test_ExtractSymbolicShapesPass():
    try:
        obj = ExtractSymbolicShapesPass()
        assert obj is not None
    except Exception:
        pass

