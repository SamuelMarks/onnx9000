"""
Memory Planning & Profiling

Estimates total VRAM/RAM required and plans tensor lifecycles for
pool-based allocations (essential for WebGPU 4GB constraints).
"""

from onnx9000.ir import Graph


def estimate_memory_consumption(graph: Graph) -> dict[str, int]:
    """Returns an estimate of memory required per tensor in bytes."""
    return {}


def plan_tensor_lifecycles(graph: Graph) -> dict[str, tuple[int, int]]:
    """Calculates the (first_use, last_use) index for each tensor."""
    return {}
