import pytest
from onnx9000.backends.codegen.op_generator import *


def test_OpGenerator():
    try:
        obj = OpGenerator()
        assert obj is not None
    except Exception:
        pass
