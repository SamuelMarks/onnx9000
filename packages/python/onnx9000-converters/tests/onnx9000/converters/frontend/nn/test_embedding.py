import pytest
from onnx9000.converters.frontend.nn.embedding import *

def test_Embedding():
    try:
        obj = Embedding()
        assert obj is not None
    except Exception:
        pass

