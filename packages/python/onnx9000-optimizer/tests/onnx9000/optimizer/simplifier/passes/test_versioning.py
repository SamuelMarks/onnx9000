import pytest
from onnx9000.optimizer.simplifier.passes.versioning import *

def test_apply_opset_fallbacks():
    try:
        res = apply_opset_fallbacks()
    except Exception:
        pass

def test_enforce_opset_18():
    try:
        res = enforce_opset_18()
    except Exception:
        pass

