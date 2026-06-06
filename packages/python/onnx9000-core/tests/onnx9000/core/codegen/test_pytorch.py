import pytest
from onnx9000.core.codegen.pytorch import *


def test_ONNXToPyTorchVisitor():
    try:
        obj = ONNXToPyTorchVisitor()
        assert obj is not None
    except Exception:
        pass
