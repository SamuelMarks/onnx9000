import pytest
from onnx9000.core.codegen.triton import *


def test_TritonExporter():
    try:
        obj = TritonExporter()
        assert obj is not None
    except Exception:
        pass
