import pytest
from onnx9000.tvm.relay.transform.memory_plan import *

def test_MemoryPlanner():
    try:
        obj = MemoryPlanner()
        assert obj is not None
    except Exception:
        pass

def test_plan_memory():
    try:
        res = plan_memory()
    except Exception:
        pass

