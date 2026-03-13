"""Module docstring."""

from onnx9000.autograd.memory import optimize_backward_memory
from onnx9000.ir import Graph, Node


def test_optimize_backward_memory():
    """test_optimize_backward_memory docstring."""
    graph = Graph("test")
    node = Node("ReluGrad", ["a", "b"], ["c"], {}, name="n")
    graph.add_node(node)

    optimize_backward_memory(graph)
    assert len(graph.nodes) == 1
