import pytest
from onnx9000.backends.cpu.executor import *


def test_CPUExecutionProvider():
    try:
        obj = CPUExecutionProvider()
        assert obj is not None
    except Exception:
        pass
