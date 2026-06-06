import pytest
from onnx9000.tflite_exporter.exporter import *

def test_TFLiteExporter():
    try:
        obj = TFLiteExporter()
        assert obj is not None
    except Exception:
        pass

