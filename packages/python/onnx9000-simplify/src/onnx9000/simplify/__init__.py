"""ONNX graph simplification, constant folding, and dead-code elimination (`simplify`)."""

from onnx9000.core.ir import Graph
from onnx9000.optimizer.simplifier import simplify


def simplify_graph(
    graph: Graph,
    skip_fusions: bool = False,
    skip_constant_folding: bool = False,
    strip_metadata: bool = False,
    max_iterations: int = 5,
) -> Graph:
    """Simplify an ONNX9000 `Graph` using iterative optimization passes.

    Args:
        graph: Input ONNX9000 `Graph`.
        skip_fusions: Whether to disable operator fusion passes.
        skip_constant_folding: Whether to disable constant folding.
        strip_metadata: Whether to strip docstrings and metadata properties.
        max_iterations: Maximum number of fixed-point simplification iterations.

    Returns:
        Simplified ONNX9000 `Graph`.
    """
    return simplify(
        graph,
        skip_fusions=skip_fusions,
        skip_constant_folding=skip_constant_folding,
        strip_metadata=strip_metadata,
        max_iterations=max_iterations,
    )


def run() -> str:
    """Execute the default simplify module status check.

    Returns:
        Simplify status confirmation string.
    """
    return "[simplify] processed"


__all__ = ["run", "simplify", "simplify_graph"]
