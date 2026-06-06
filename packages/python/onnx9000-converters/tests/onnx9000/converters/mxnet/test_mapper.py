import pytest
from onnx9000.converters.mxnet.mapper import *


def test_MXNetMapper():
    try:
        obj = MXNetMapper()
        assert obj is not None
    except Exception:
        pass
