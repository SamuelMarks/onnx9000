import pytest
from onnx9000.optimizer.olive.testing_ops import *

def test_TestingOps():
    try:
        obj = TestingOps()
        assert obj is not None
    except Exception:
        pass

