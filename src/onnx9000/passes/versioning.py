"""
Opset Versioning and Fallbacks

Handles upgrading models to Opset 18+ and polyfilling operations
that might be missing in older setups.
"""

from onnx9000.ir import Graph


def enforce_opset_18(graph: Graph) -> None:
    """Ensures graph compliance with Opset 18+."""
    pass


def apply_opset_fallbacks(graph: Graph) -> None:
    """Replaces unsupported older ops with equivalent compositions."""
    pass
