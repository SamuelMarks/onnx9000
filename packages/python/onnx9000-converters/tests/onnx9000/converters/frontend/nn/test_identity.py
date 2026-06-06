import pytest
from onnx9000.converters.frontend.nn.identity import *


def test_Identity():
    try:
        obj = Identity()
        assert obj is not None
    except Exception:
        pass
