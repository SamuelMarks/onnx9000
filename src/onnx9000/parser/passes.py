"""
Optimization Pipeline Wrapper
"""

from onnx9000.ir import Graph
from onnx9000.passes import (
    constant_folding,
    fuse_consecutive_transpose,
    fuse_matmul_add,
)


def optimize(graph: Graph) -> None:
    """Runs a standard sequence of optimizations."""
    constant_folding(graph)
    fuse_consecutive_transpose(graph)
    fuse_matmul_add(graph)

    # Re-plan memory since the graph structure has changed
    from onnx9000.parser.memory import plan_memory

    plan_memory(graph)
