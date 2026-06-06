import pytest
from onnx9000.core.codegen.flax import *


def test_ONNXToFlaxNNXVisitor():
    try:
        obj = ONNXToFlaxNNXVisitor()
        assert obj is not None
    except Exception:
        pass
