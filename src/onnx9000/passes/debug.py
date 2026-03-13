"""
Debugging Passes

Injects probes (Print or Identity nodes) into the graph to extract
intermediate tensor values during execution.
"""

from onnx9000.ir import Graph


def inject_probes(graph: Graph, node_names: list[str]) -> None:
    """Injects debug probes into specific points in the graph."""
    pass
