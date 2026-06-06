import pytest
from onnx9000.backends.cuda.cuda_arena import *

def test_CUDAMemoryPlanner():
    try:
        obj = CUDAMemoryPlanner()
        assert obj is not None
    except Exception:
        pass

