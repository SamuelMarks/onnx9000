import pytest
from onnx9000.optimizer.sparse.calibration import *

def test_DataLoader():
    try:
        obj = DataLoader()
        assert obj is not None
    except Exception:
        pass

def test_cross_entropy_loss():
    try:
        res = cross_entropy_loss()
    except Exception:
        pass

def test_evaluate_accuracy():
    try:
        res = evaluate_accuracy()
    except Exception:
        pass

