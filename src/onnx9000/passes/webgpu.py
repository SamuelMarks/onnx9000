"""
WebGPU Specific Optimizations

Modifies the graph to respect WebGPU specific constraints, such as max
bind groups, storage buffer limits, and unsupported operators.
"""

from onnx9000.ir import Graph


def optimize_for_webgpu(graph: Graph) -> None:
    """Applies all WebGPU specific constraints and limits."""
    pass


def polyfill_webgpu_unsupported(graph: Graph) -> None:
    """Replaces ops that lack a WGSL implementation with polyfills."""
    pass
