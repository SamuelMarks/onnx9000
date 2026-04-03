import pytest
from unittest.mock import MagicMock, patch
from onnx9000.tensorrt.ops_matmul import trt_matmul


def test_matmul_fail():
    class MockNode:
        def __init__(self):
            self.inputs = ["in1", "in2"]
            self.outputs = ["out"]
            self.attributes = {}

    class MockTensor:
        ptr = 123

    node = MockNode()
    tensors = {"in1": MockTensor(), "in2": MockTensor()}
    network = MagicMock()
    network.ptr = 456

    mock_ffi = MagicMock()
    mock_ffi.lib = MagicMock()
    mock_ffi.lib.addMatrixMultiply.return_value = None

    with patch("onnx9000.tensorrt.ops_matmul.ffi", mock_ffi):
        with pytest.raises(RuntimeError):
            trt_matmul(network, node, tensors)
