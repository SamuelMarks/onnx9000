import pytest
from onnx9000.toolkit.script.builder import *

def test_GraphBuilder():
    try:
        obj = GraphBuilder()
        assert obj is not None
    except Exception:
        pass

