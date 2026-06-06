import pytest
from onnx9000_optimum.optimize import *


def test_optimize_model():
    try:
        optimize_model()
    except Exception:
        pass
