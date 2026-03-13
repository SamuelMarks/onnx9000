"""
Autograd Utilities

Helper functions for graph traversal and manipulation during backward pass construction.
"""

from onnx9000.ir import Graph, Node


def reverse_topological_sort(graph: Graph) -> list[Node]:
    """
    Returns the nodes of the graph in reverse topological order.
    Assuming the graph nodes are already in topological order (from ONNX),
    this is a simple reversal.
    """
    return list(reversed(graph.nodes))
