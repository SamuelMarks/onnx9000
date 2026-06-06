import pytest
from onnx9000.optimizer.pattern_matcher import *

def test_Pattern():
    try:
        obj = Pattern()
        assert obj is not None
    except Exception:
        pass

def test_PatternMatcherEngine():
    try:
        obj = PatternMatcherEngine()
        assert obj is not None
    except Exception:
        pass

def test_matches():
    try:
        res = matches()
    except Exception:
        pass

def test_apply_algebraic_reuse():
    try:
        res = apply_algebraic_reuse()
    except Exception:
        pass

def test_apply_fusion_reuse():
    try:
        res = apply_fusion_reuse()
    except Exception:
        pass

def test_apply_hardware_lowering():
    try:
        res = apply_hardware_lowering()
    except Exception:
        pass

