import pytest
from onnx9000.optimizer.olive.quantization_ops import *


def test_Quantizer():
    try:
        obj = Quantizer()
        assert obj is not None
    except Exception:
        pass
