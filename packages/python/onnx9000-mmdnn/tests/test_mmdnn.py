"""Comprehensive unit tests for onnx9000.mmdnn and onnx9000_mmdnn."""

import json

import onnx9000_mmdnn
import pytest
from onnx9000.core.ir import Graph
from onnx9000.mmdnn import SUPPORTED_FRAMEWORKS, MMDNNConverter, run


def test_mmdnn_converter_and_run() -> None:
    """Verify run(), MMDNNConverter validation, and CNTK JSON bytes conversion."""
    assert run() == "[mmdnn] processed"
    assert onnx9000_mmdnn.run() == "[mmdnn] processed"
    assert "caffe" in SUPPORTED_FRAMEWORKS

    with pytest.raises(ValueError, match="Unsupported MMDNN framework"):
        MMDNNConverter("unknown_fw")

    conv = MMDNNConverter("cntk")
    cntk_payload = json.dumps({"nodes": [{"op": "ReLU", "name": "relu1", "inputs": ["x"]}]}).encode(
        "utf-8"
    )
    graph = conv.convert_cntk_bytes(cntk_payload)
    assert isinstance(graph, Graph)
