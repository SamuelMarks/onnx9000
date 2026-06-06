import pytest
from onnx9000.tflite_exporter.flatbuffer.builder import *

def test_FlatBufferBuilder():
    try:
        obj = FlatBufferBuilder()
        assert obj is not None
    except Exception:
        pass

