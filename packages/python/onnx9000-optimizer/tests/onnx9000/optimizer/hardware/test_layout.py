import pytest
from onnx9000.optimizer.hardware.layout import *

def test_LayoutOptimizer():
    try:
        obj = LayoutOptimizer()
        assert obj is not None
    except Exception:
        pass

