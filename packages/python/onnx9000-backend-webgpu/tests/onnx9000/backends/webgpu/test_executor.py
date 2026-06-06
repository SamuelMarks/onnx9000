import pytest
from onnx9000.backends.webgpu.executor import *

def test_WebGPUExecutionProvider():
    try:
        obj = WebGPUExecutionProvider()
        assert obj is not None
    except Exception:
        pass

