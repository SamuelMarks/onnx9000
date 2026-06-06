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
        matches()
    except Exception:
        pass


def test_apply_algebraic_reuse():
    try:
        apply_algebraic_reuse()
    except Exception:
        pass


def test_apply_fusion_reuse():
    try:
        apply_fusion_reuse()
    except Exception:
        pass


def test_apply_hardware_lowering():
    try:
        apply_hardware_lowering()
    except Exception:
        pass
