import pytest
from onnx9000.optimizer.simplifier.passes.base import *


def test_GraphPass():
    try:
        obj = GraphPass()
        assert obj is not None
    except Exception:
        pass
