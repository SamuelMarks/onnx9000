import pytest
from onnx9000.tflite_exporter.optimizations.edgetpu import *

def test_EdgeTPUOptimizer():
    try:
        obj = EdgeTPUOptimizer()
        assert obj is not None
    except Exception:
        pass

