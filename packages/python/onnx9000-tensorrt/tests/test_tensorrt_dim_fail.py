"""Tests for tensorrt dim fail."""

from unittest.mock import MagicMock, patch

import pytest
from onnx9000.tensorrt.ops_dim import trt_reshape


def test_reshape_missing_input():
    """Docstring for D103."""

    class MockNode:
        """Mock node."""

        def __init__(self):
            """Init."""
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
