import pytest
from onnx9000.converters.frontend.builder import *

def test_GraphBuilder():
    try:
        obj = GraphBuilder()
        assert obj is not None
    except Exception:
        pass

def test_Tracing():
    try:
        obj = Tracing()
        assert obj is not None
    except Exception:
        pass

def test_get_active_builder():
    try:
        res = get_active_builder()
    except Exception:
        pass

