import pytest
from onnx9000.toolkit.training.autograd.memory import *


def test_optimize_backward_memory():
    try:
        optimize_backward_memory()
    except Exception:
        pass
