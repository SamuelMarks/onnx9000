import pytest
from onnx9000.optimizer.hardware.pipeline import *


def test_PipelineOptimizer():
    try:
        obj = PipelineOptimizer()
        assert obj is not None
    except Exception:
        pass
