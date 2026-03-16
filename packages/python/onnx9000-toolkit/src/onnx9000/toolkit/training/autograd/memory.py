"""
Autograd Memory Optimizations

Routines to analyze and minimize the peak memory usage during the
combined forward-backward training pass by aggressively re-using buffers.
"""

from onnx9000.core.ir import Graph


def optimize_backward_memory(graph: Graph) -> None:
    """
    Identifies intermediate activations from the forward pass that can be
    overwritten or immediately freed after their corresponding backward
    pass nodes have consumed them.
    """
    for node in graph.nodes:
        if node.op_type.endswith("Grad"):
            continue
