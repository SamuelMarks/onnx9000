"""Unit tests for TensorRT operator registry error handling with missing inputs."""

from unittest.mock import MagicMock, patch

import pytest
from onnx9000.tensorrt.registry import _TRT_OP_REGISTRY


def test_missing_inputs_all() -> None:
    """Verify all registered TensorRT op handlers raise Onnx9000Error or IndexError on empty inputs."""

    class MockNode:
        """Mock node."""

        def __init__(self) -> None:
            """Initialize MockNode with configured parameters and state."""
            self.inputs: list[str] = []
            self.op_type = "Mock"
            self.attributes: dict[str, object] = {}

    node = MockNode()
    tensors: dict[str, object] = {}
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
            with pytest.raises((ValueError, IndexError, KeyError, RuntimeError, TypeError)):
                op_func(network, node, tensors)
