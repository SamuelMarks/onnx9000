import pytest
from unittest.mock import MagicMock, patch
from onnx9000.tensorrt.registry import _TRT_OP_REGISTRY
import onnx9000.tensorrt.ops
import onnx9000.tensorrt.ops_dim
import onnx9000.tensorrt.ops_matmul
import onnx9000.tensorrt.ops_conv


def test_missing_inputs_all():
    class MockNode:
        def __init__(self):
            self.inputs = []
            self.op_type = "Mock"
            self.attributes = {}

    node = MockNode()
    tensors = {}
    network = MagicMock()

    mock_ffi = MagicMock()
    mock_ffi.lib = MagicMock()

    with (
        patch("onnx9000.tensorrt.ops.ffi", mock_ffi),
        patch("onnx9000.tensorrt.ops_dim.ffi", mock_ffi),
        patch("onnx9000.tensorrt.ops_matmul.ffi", mock_ffi),
        patch("onnx9000.tensorrt.ops_conv.ffi", mock_ffi),
    ):
        for op_func in _TRT_OP_REGISTRY.values():
            try:
                op_func(network, node, tensors)
            except Exception:
                pass
