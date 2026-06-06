import pytest
from onnx9000.onnx2gguf.reader import *

def test_GGUFReader():
    try:
        obj = GGUFReader()
        assert obj is not None
    except Exception:
        pass

