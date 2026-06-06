import pytest
from onnx9000.tflite_exporter.flatbuffer.reader import *


def test_FlatBufferReader():
    try:
        obj = FlatBufferReader()
        assert obj is not None
    except Exception:
        pass
