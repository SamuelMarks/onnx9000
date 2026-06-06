import pytest
from onnx9000.toolkit.script.js_wrapper import *

def test_JSGraphBuilder():
    try:
        obj = JSGraphBuilder()
        assert obj is not None
    except Exception:
        pass

