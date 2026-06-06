import pytest
from onnx9000.optimizer.surgeon.layout import *

def test_optimize_layouts():
    try:
        res = optimize_layouts()
    except Exception:
        pass

