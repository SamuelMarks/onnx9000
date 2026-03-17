"""Tests the ir coverage module functionality."""

from onnx9000.core.ir import Graph


def test_uniquify_empty() -> None:
    """Tests the uniquify empty functionality."""
    g = Graph("test")
    name = g._uniquify_node_name("")
    assert name == "node"
