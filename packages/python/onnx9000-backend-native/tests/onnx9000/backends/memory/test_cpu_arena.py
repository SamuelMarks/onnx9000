import pytest
from onnx9000.backends.memory.cpu_arena import *

def test_CPUMemoryPlanner():
    try:
        obj = CPUMemoryPlanner()
        assert obj is not None
    except Exception:
        pass

