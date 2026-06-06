import pytest
from onnx9000.core.onnx_pb2 import *

def test_module_load():
    import onnx9000.core.onnx_pb2
    assert onnx9000.core.onnx_pb2 is not None

