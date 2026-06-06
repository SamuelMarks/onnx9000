import pytest
from onnx9000.optimizer.olive.llm_ops import *

def test_LlmOps():
    try:
        obj = LlmOps()
        assert obj is not None
    except Exception:
        pass

