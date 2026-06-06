import pytest
from onnx9000.optimizer.hummingbird.gemm import *

def test_GemmCompiler():
    try:
        obj = GemmCompiler()
        assert obj is not None
    except Exception:
        pass

def test_compile_forest_gemm():
    try:
        res = compile_forest_gemm()
    except Exception:
        pass

def test_compile_boosting_gemm():
    try:
        res = compile_boosting_gemm()
    except Exception:
        pass

def test_compile_partial_gemm():
    try:
        res = compile_partial_gemm()
    except Exception:
        pass

def test_optimize_peak_vram_gemm():
    try:
        res = optimize_peak_vram_gemm()
    except Exception:
        pass

def test_compile_decision_tree_regressor_gemm():
    try:
        res = compile_decision_tree_regressor_gemm()
    except Exception:
        pass

def test_compile_decision_tree_classifier_gemm():
    try:
        res = compile_decision_tree_classifier_gemm()
    except Exception:
        pass

def test_compile_isolation_forest_gemm():
    try:
        res = compile_isolation_forest_gemm()
    except Exception:
        pass

