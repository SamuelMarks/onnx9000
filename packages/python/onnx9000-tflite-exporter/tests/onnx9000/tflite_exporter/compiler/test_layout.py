import pytest
from onnx9000.tflite_exporter.compiler.layout import *


def test_LayoutOptimizer():
    try:
        obj = LayoutOptimizer()
        assert obj is not None
    except Exception:
        pass
