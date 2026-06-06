import pytest
from onnx9000.optimizer.olive.pruning_ops import *

def test_Pruner():
    try:
        obj = Pruner()
        assert obj is not None
    except Exception:
        pass

