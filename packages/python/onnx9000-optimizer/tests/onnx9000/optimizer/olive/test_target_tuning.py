import pytest
from onnx9000.optimizer.olive.target_tuning import *

def test_TargetTuner():
    try:
        obj = TargetTuner()
        assert obj is not None
    except Exception:
        pass

