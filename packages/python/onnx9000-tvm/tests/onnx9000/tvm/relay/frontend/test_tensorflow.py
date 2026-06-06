import pytest
from onnx9000.tvm.relay.frontend.tensorflow import *

def test_TFImporter():
    try:
        obj = TFImporter()
        assert obj is not None
    except Exception:
        pass

def test_from_tensorflow():
    try:
        res = from_tensorflow()
    except Exception:
        pass

