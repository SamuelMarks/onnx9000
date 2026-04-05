"""Tests for tensorrt matmul fail."""

from unittest.mock import MagicMock, patch

import pytest
from onnx9000.tensorrt.ops_matmul import trt_matmul


def test_matmul_fail():
    """Docstring for D103."""

    class MockNode:
        """Mock node."""

        def __init__(self):
            """Init."""
            self.inputs = ["in1", "in2"]
            self.outputs = ["out"]
            self.attributes = {}

    class MockTensor:
        """Mock tensor."""

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
