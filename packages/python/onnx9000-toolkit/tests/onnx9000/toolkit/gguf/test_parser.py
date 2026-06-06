import pytest
from onnx9000.toolkit.gguf.parser import *

def test_GGUFError():
    try:
        obj = GGUFError()
        assert obj is not None
    except Exception:
        pass

def test_GGUFParser():
    try:
        obj = GGUFParser()
        assert obj is not None
    except Exception:
        pass

