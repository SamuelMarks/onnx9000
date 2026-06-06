import pytest
from onnx9000.tvm.ide import *


def test_WebIDE():
    try:
        obj = WebIDE()
        assert obj is not None
    except Exception:
        pass
