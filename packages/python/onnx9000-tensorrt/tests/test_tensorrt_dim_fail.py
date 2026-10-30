"""Tests for tensorrt dim fail."""

from unittest.mock import MagicMock, patch

import pytest
from onnx9000.tensorrt.ops_dim import trt_reshape


def test_reshape_missing_input() -> None:
    """Verify that reshape missing input behaves as expected."""

    class MockNode:
        """Mock node."""

        def __init__(self) -> None:
            """Initialize MockNode with configured parameters and state."""
            self.inputs = []
            self.outputs = ["out"]

    node = MockNode()
    tensors = {}
    network = MagicMock()

    mock_ffi = MagicMock()
    mock_ffi.lib = MagicMock()

    with patch("onnx9000.tensorrt.ops_dim.ffi", mock_ffi):
        with pytest.raises(RuntimeError, match="Missing input"):
            trt_reshape(network, node, tensors)
