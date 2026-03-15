"""Module providing core logic and structural definitions."""

from onnx9000.training.autograd.compiler import extract_partial_subgraph
from onnx9000.core.ir import Graph, Node


def test_extract_partial_subgraph():
    """Provides semantic functionality and verification."""
    g = Graph("test")
    g.add_node(Node("Identity", ["a"], ["b"], {}, name="test_node"))
    sub = extract_partial_subgraph(g, [], [])
    assert sub.name == "test_partial"
    assert len(sub.nodes) == 1
