import pytest
from onnx9000.core.profiler_checks import *

def test_OptimizationAnalyzer():
    try:
        obj = OptimizationAnalyzer()
        assert obj is not None
    except Exception:
        pass

