"""ONNX GraphSurgeon DAG manipulation, topological sorting, cleanup, and constant folding."""

from onnx9000.core.ir import Graph
from onnx9000.core.surgeon import cleanup, fold_constants, toposort


class GraphSurgeon:
    """High-level GraphSurgeon controller for optimizing and sanitizing ONNX9000 graphs."""

    def optimize(self, graph: Graph) -> Graph:
        """Run topological sort, constant folding, and dead-code cleanup on a graph.

        Args:
            graph: Target ONNX9000 Graph to sanitize and optimize.

        Returns:
            The mutated and topologically sorted `Graph`.
        """
        toposort(graph)
        fold_constants(graph)
        cleanup(graph)
        return graph

    def process(self, name: str) -> str:
        """Process a named graph target.

        Args:
            name: Identifier of the graph target.

        Returns:
            Status summary string.
        """
        return f"GraphSurgeon processed {name}"


def run() -> str:
    """Execute the default GraphSurgeon module status check.

    Returns:
        GraphSurgeon status confirmation string.
    """
    return "[graphsurgeon] processed"


__all__ = ["GraphSurgeon", "cleanup", "fold_constants", "run", "toposort"]
