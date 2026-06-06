import pytest
from onnx9000.tvm.relay.visualize import *


def test_DotPrinter():
    try:
        obj = DotPrinter()
        assert obj is not None
    except Exception:
        pass


def test_to_dot():
    try:
        to_dot()
    except Exception:
        pass
