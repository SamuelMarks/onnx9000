"""Provides validation.py module functionality."""

import logging
from typing import Set

from onnx9000.core.ir import Graph, Node
from onnx9000.optimize.simplifier.passes.base import GraphPass

logger = logging.getLogger(__name__)


class ValidationPass(GraphPass):
    """
    Graph Validation.
    Sanity checks for graph structures, such as detecting cycles or
    disconnected subgraphs.
    """

    def run(self, graph: Graph) -> bool:
        """Provides run functionality and verification."""
        self.detect_cycles(graph)
        self.detect_dangling(graph)
        return False

    def detect_cycles(self, graph: Graph) -> None:
        """
        Performs a topological sort or DFS to ensure the graph is a strict DAG.
        Raises an error if a cycle is detected.
        """
        adj = {node.name: [] for node in graph.nodes}

        # Build adjacency list
        node_by_output = {}
        for node in graph.nodes:
            for out in node.outputs:
                node_by_output[out] = node

        for node in graph.nodes:
            for inp in node.inputs:
                if inp in node_by_output:
                    adj[node_by_output[inp].name].append(node.name)

        visited = set()
        rec_stack = set()

        def is_cyclic(u: str) -> bool:
            """Provides is cyclic functionality and verification."""
            visited.add(u)
            rec_stack.add(u)
            for neighbor in adj.get(u, []):
                if neighbor not in visited:
                    if is_cyclic(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(u)
            return False

        for node in graph.nodes:
            if node.name not in visited:
                if is_cyclic(node.name):
                    raise RuntimeError(
                        f"Cycle detected in graph starting from node {node.name}"
                    )

    def detect_dangling(self, graph: Graph) -> None:
        """Provides detect dangling functionality and verification."""
        available = set(graph.inputs) | set(graph.initializers)
        for n in graph.nodes:
            for out in n.outputs:
                available.add(out)

        for n in graph.nodes:
            for inp in n.inputs:
                if inp and inp not in available:
                    logger.warning(f"Dangling input detected: {inp} in node {n.name}")


def detect_cycles(graph: Graph) -> None:
    """Provides detect cycles functionality and verification."""
    ValidationPass().detect_cycles(graph)


def detect_dangling(graph: Graph) -> None:
    """Provides detect dangling functionality and verification."""
    ValidationPass().detect_dangling(graph)
