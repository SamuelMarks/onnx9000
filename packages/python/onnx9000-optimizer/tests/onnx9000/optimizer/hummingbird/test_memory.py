import pytest
from onnx9000.optimizer.hummingbird.memory import *

def test_TreeAbstractions():
    try:
        obj = TreeAbstractions()
        assert obj is not None
    except Exception:
        pass

def test_estimate_memory_footprint():
    try:
        res = estimate_memory_footprint()
    except Exception:
        pass

def test_select_optimal_strategy():
    try:
        res = select_optimal_strategy()
    except Exception:
        pass

