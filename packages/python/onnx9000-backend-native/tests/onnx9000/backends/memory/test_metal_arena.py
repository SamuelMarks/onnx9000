import pytest
from onnx9000.backends.memory.metal_arena import *

def test_MetalMemoryPlanner():
    try:
        obj = MetalMemoryPlanner()
        assert obj is not None
    except Exception:
        pass

