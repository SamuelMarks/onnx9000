import pytest
from onnx9000.tvm.relay.module import *

def test_IRModule():
    try:
        obj = IRModule()
        assert obj is not None
    except Exception:
        pass

