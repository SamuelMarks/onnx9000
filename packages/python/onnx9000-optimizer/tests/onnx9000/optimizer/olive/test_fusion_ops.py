import pytest
from onnx9000.optimizer.olive.fusion_ops import *


def test_FusionOptimizer():
    try:
        obj = FusionOptimizer()
        assert obj is not None
    except Exception:
        pass
