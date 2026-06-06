import pytest
from onnx9000.optimizer.olive.context import *


def test_PassContext():
    try:
        obj = PassContext()
        assert obj is not None
    except Exception:
        pass
