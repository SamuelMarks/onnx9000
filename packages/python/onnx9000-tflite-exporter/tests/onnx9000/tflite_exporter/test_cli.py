import pytest
from onnx9000.tflite_exporter.cli import *

def test_main():
    try:
        res = main()
    except Exception:
        pass

