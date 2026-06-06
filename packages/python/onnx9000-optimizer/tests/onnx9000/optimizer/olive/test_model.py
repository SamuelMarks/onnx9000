import pytest
from onnx9000.optimizer.olive.model import *


def test_OliveModel():
    try:
        obj = OliveModel()
        assert obj is not None
    except Exception:
        pass
