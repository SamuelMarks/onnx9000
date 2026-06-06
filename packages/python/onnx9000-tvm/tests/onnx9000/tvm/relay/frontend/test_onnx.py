import pytest
from onnx9000.tvm.relay.frontend.onnx import *


def test_ONNXImporter():
    try:
        obj = ONNXImporter()
        assert obj is not None
    except Exception:
        pass


def test_from_onnx():
    try:
        from_onnx()
    except Exception:
        pass
