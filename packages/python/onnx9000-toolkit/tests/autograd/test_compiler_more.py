"""Module providing core logic and structural definitions."""

from onnx9000.core.ir import Graph, Node
from onnx9000.toolkit.training.autograd.compiler import extract_partial_subgraph


def test_extract_partial_subgraph() -> None:
    """Tests the test_extract_partial_subgraph functionality."""
    g = Graph("test")
    g.add_node(Node("Identity", ["a"], ["b"], {}, name="test_node"))
    sub = extract_partial_subgraph(g, [], [])
    assert sub.name == "test_partial"
    assert len(sub.nodes) == 1
