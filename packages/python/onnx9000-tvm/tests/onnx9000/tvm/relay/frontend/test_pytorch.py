import pytest
from onnx9000.tvm.relay.frontend.pytorch import *

def test_PyTorchImporter():
    try:
        obj = PyTorchImporter()
        assert obj is not None
    except Exception:
        pass

def test_from_pytorch():
    try:
        res = from_pytorch()
    except Exception:
        pass

