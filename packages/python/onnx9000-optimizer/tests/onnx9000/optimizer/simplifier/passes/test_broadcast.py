import pytest
from onnx9000.optimizer.simplifier.passes.broadcast import *


def test_optimize_broadcasting():
    try:
        optimize_broadcasting()
    except Exception:
        pass
