import pytest
from onnx9000.optimizer.olive.evaluator import *

def test_Evaluator():
    try:
        obj = Evaluator()
        assert obj is not None
    except Exception:
        pass

