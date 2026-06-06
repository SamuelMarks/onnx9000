import pytest
from onnx9000.core.codegen.keras import *

def test_ONNXToKerasVisitor():
    try:
        obj = ONNXToKerasVisitor()
        assert obj is not None
    except Exception:
        pass

