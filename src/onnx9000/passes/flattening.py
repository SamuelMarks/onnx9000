"""
Graph Flattening

Flattens nested control flow (If, Loop) into standard linear execution
traces where possible, or unrolls static loops.
"""

from onnx9000.ir import Graph


def flatten_subgraphs(graph: Graph) -> None:
    """Flattens static nested subgraphs into the main graph."""
    pass
