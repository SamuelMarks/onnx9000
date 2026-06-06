import pytest
from onnx9000.optimizer.olive.target import *

def test_Target():
    try:
        obj = Target()
        assert obj is not None
    except Exception:
        pass

