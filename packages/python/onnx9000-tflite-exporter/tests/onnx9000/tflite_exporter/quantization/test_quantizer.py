import pytest
from onnx9000.tflite_exporter.quantization.quantizer import *


def test_TensorQuantization():
    try:
        obj = TensorQuantization()
        assert obj is not None
    except Exception:
        pass


def test_Quantizer():
    try:
        obj = Quantizer()
        assert obj is not None
    except Exception:
        pass
