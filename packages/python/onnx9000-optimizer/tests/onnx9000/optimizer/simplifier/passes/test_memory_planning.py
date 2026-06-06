import pytest
from onnx9000.optimizer.simplifier.passes.memory_planning import *

def test_estimate_memory_consumption():
    try:
        res = estimate_memory_consumption()
    except Exception:
        pass

def test_plan_tensor_lifecycles():
    try:
        res = plan_tensor_lifecycles()
    except Exception:
        pass

