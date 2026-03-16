"""Target hardware definitions."""

from enum import Enum, auto


class Target(Enum):
    """Target hardware platforms."""

    CPU = auto()
    WebGPU = auto()
    WASM_SIMD = auto()
    Accelerate = auto()
    CoreML = auto()
    TensorRT = auto()
