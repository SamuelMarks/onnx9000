"""
Constant Folding

Evaluates deterministic operations at compile-time when all of their
inputs are statically known constants or initializers.
"""

from onnx9000.ir import Graph


def constant_folding(graph: Graph) -> None:
    """Folds constants in the graph."""
    # Full implementation requires a mini-interpreter to execute nodes.
    # For now, we stub the interface for the architecture.
    pass
