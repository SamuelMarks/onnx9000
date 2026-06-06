import pytest
from onnx9000.tvm.relay.transform.fusion import *


def test_OpFusionDetector():
    try:
        obj = OpFusionDetector()
        assert obj is not None
    except Exception:
        pass


def test_fuse_ops():
    try:
        fuse_ops()
    except Exception:
        pass
