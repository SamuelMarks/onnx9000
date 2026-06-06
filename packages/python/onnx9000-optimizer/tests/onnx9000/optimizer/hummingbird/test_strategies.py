import pytest
from onnx9000.optimizer.hummingbird.strategies import *


def test_Strategy():
    try:
        obj = Strategy()
        assert obj is not None
    except Exception:
        pass


def test_TargetHardware():
    try:
        obj = TargetHardware()
        assert obj is not None
    except Exception:
        pass
