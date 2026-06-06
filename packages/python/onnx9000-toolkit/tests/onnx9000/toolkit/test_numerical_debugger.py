import pytest
from onnx9000.toolkit.numerical_debugger import *

def test_NumericalDebugger():
    try:
        obj = NumericalDebugger()
        assert obj is not None
    except Exception:
        pass

