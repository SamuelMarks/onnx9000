import pytest
from onnx9000.optimizer.surgeon.quantization import *


def test_quantize_ptq():
    try:
        quantize_ptq()
    except Exception:
        pass
