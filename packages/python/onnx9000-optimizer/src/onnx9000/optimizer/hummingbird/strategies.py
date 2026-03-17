from enum import Enum, auto


class Strategy(Enum):
    """Execution strategy for traditional ML models."""

    GEMM = auto()
    TREE_TRAVERSAL = auto()
    PERFECT_TREE_TRAVERSAL = auto()


class TargetHardware(Enum):
    """Hardware target for execution."""

    CPU = auto()
    GPU = auto()
    WEBGPU = auto()
