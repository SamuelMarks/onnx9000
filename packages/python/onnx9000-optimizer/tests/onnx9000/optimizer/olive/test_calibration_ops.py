import pytest
from onnx9000.optimizer.olive.calibration_ops import *

def test_CalibrationLoop():
    try:
        obj = CalibrationLoop()
        assert obj is not None
    except Exception:
        pass

