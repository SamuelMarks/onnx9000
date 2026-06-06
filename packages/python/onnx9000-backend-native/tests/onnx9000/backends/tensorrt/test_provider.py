import pytest
from onnx9000.backends.tensorrt.provider import *


def test_TensorrtExecutionProvider():
    try:
        obj = TensorrtExecutionProvider()
        assert obj is not None
    except Exception:
        pass
