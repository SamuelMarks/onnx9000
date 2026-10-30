"""Provides validation.py module functionality."""

import logging

from onnx9000.core.exceptions import OptimizerError
from onnx9000.core.ir import Graph
from onnx9000.optimizer.simplifier.passes.base import GraphPass

logger = logging.getLogger(__name__)


class ValidationPass(GraphPass):
    """Graph Validation.

    Sanity checks for graph structures, such as detecting cycles or
    disconnected subgraphs.
    """

    def run(self, graph: Graph) -> bool:
        """Execute run on ValidationPass.

        Args:
            graph: Parameter specifying graph (Graph).

        Returns:
            Result of run (bool).
        """
        self.detect_cycles(graph)
        self.detect_dangling(graph)
        return False

    def detect_cycles(self, graph: Graph) -> None:
        """Perform a topological sort or DFS to ensure the graph is a strict DAG.

        Raises an error if a cycle is detected.
        """
        adj = {node.name: [] for node in graph.nodes}
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
            """Execute is cyclic operation.

            Args:
                u: Parameter specifying u (str).

            Returns:
                Result of is cyclic (bool).
            """
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
            if node.name not in visited and is_cyclic(node.name):
                raise OptimizerError(f"Cycle detected in graph starting from node {node.name}")

    def detect_dangling(self, graph: Graph) -> None:
        """Execute detect dangling on ValidationPass.

        Args:
            graph: Parameter specifying graph (Graph).
        """
        available = set(graph.inputs) | set(graph.initializers)
        for n in graph.nodes:
            for out in n.outputs:
                available.add(out)
        for n in graph.nodes:
            for inp in n.inputs:
                if inp and inp not in available:
                    logger.warning(f"Dangling input detected: {inp} in node {n.name}")


def detect_cycles(graph: Graph) -> None:
    """Execute detect cycles operation.

    Args:
        graph: Parameter specifying graph (Graph).
    """
    ValidationPass().detect_cycles(graph)


def detect_dangling(graph: Graph) -> None:
    """Execute detect dangling operation.

    Args:
        graph: Parameter specifying graph (Graph).
    """
    ValidationPass().detect_dangling(graph)
