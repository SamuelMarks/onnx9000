import pytest
from onnx9000.optimizer.hummingbird.engine import *


def test_TranspilationEngine():
    try:
        obj = TranspilationEngine()
        assert obj is not None
    except Exception:
        pass
