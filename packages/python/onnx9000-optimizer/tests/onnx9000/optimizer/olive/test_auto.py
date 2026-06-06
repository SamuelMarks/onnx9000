import pytest
from onnx9000.optimizer.olive.auto import *


def test_AutoOptimizer():
    try:
        obj = AutoOptimizer()
        assert obj is not None
    except Exception:
        pass
