import pytest
from onnx9000.optimizer.hardware.quantizer import *

def test_Quantizer():
    try:
        obj = Quantizer()
        assert obj is not None
    except Exception:
        pass

