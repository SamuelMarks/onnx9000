"""TVM submodule for AST and optimization."""

from .schedule import Schedule, create_schedule
from .tensor import Tensor


def default_x86_schedule(ops) -> Schedule:
    """Create default schedule for x86 CPUs."""
    s = create_schedule(ops)
    # apply default x86 heuristics (e.g. vectorization)
    return s


def default_arm_schedule(ops) -> Schedule:
    """Create default schedule for ARM CPUs."""
    s = create_schedule(ops)
    return s


def default_wasm_schedule(ops) -> Schedule:
    """Create default schedule for WebAssembly (SIMD v128)."""
    s = create_schedule(ops)
    return s


def default_webgpu_schedule(ops) -> Schedule:
    """Create default schedule for WebGPU (Workgroups)."""
    s = create_schedule(ops)
    return s
