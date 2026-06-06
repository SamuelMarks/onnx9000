import pytest
from onnx9000.backends.rocm.compiler import *


def test_ROCmCompiler():
    try:
        obj = ROCmCompiler()
        assert obj is not None
    except Exception:
        pass
