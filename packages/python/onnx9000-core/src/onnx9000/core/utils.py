"""Use utility functions for topological sorting and DAG manipulations."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.core.ir import Graph, Node


class CyclicDependencyError(Onnx9000Error):
    """Exception raised when a cyclic dependency is detected during topological sort."""


def topological_sort(graph: Graph) -> list[Node]:
    """Perform a topological sort of the nodes in the graph.

    Throws CyclicDependencyError if a cycle is detected.
    """
    producer_map: dict[str, Node] = {}
    for node in graph.nodes:
        for output_name in node.outputs:
            producer_map[output_name] = node
    visited: set[Node] = set()
    visiting: set[Node] = set()
    sorted_nodes: list[Node] = []

    def visit(node: Node) -> None:
        """Visit function logic implementation."""
        if node in visiting:
            raise CyclicDependencyError(
                f"Cycle detected involving node {node.name or node.op_type}"
            )
        if node in visited:
            return
        visiting.add(node)
        for input_name in node.inputs:
            if input_name in producer_map:
                visit(producer_map[input_name])
        visiting.remove(node)
        visited.add(node)
        sorted_nodes.append(node)

    for node in graph.nodes:
        if node not in visited:
            visit(node)
    return sorted_nodes
