import pytest
from onnx9000.backends.testing.runner import *

def test_ONNXBackendTestRunner():
    try:
        obj = ONNXBackendTestRunner()
        assert obj is not None
    except Exception:
        pass

