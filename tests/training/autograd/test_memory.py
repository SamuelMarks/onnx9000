"""Module providing core logic and structural definitions."""

from onnx9000.training.autograd.memory import optimize_backward_memory
from onnx9000.core.ir import Graph, Node


def test_optimize_backward_memory():
    """Tests the test optimize backward memory functionality."""
    graph = Graph("test")
    node = Node("ReluGrad", ["a", "b"], ["c"], {}, name="n")
    graph.add_node(node)
    optimize_backward_memory(graph)
    assert len(graph.nodes) == 1
