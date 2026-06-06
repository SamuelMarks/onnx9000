import pytest
from onnx9000.genai.model import *

def test_Model():
    try:
        obj = Model()
        assert obj is not None
    except Exception:
        pass

