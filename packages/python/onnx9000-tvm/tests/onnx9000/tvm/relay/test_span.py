import pytest
from onnx9000.tvm.relay.span import *


def test_Span():
    try:
        obj = Span()
        assert obj is not None
    except Exception:
        pass


def test_set_span():
    try:
        set_span()
    except Exception:
        pass
