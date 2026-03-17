import logging
from typing import Any, Dict, List, Optional
from onnx9000.optimizer.hummingbird.strategies import Strategy, TargetHardware
from onnx9000.core.ir import Graph, Node, Tensor

logger = logging.getLogger(__name__)


class TreeAbstractions:
    """Intermediate abstractions mapping tree nodes to native tensors."""

    def __init__(self):
        self.features: list[int] = []
        self.thresholds: list[float] = []
        self.left_children: list[int] = []
        self.right_children: list[int] = []
        self.values: list[float] = []
        self.missing_tracks: list[int] = []

    def add_node(
        self, feature: int, threshold: float, left: int, right: int, value: float, missing: int = 0
    ):
        self.features.append(feature)
        self.thresholds.append(threshold)
        self.left_children.append(left)
        self.right_children.append(right)
        self.values.append(value)
        self.missing_tracks.append(missing)


def estimate_memory_footprint(
    abstractions: TreeAbstractions, strategy: Strategy, batch_size: int = 1
) -> int:
    """Implement memory-footprint estimator to auto-select optimal strategy.
    Returns estimated bytes.
    """
    num_nodes = len(abstractions.features)
    if strategy == Strategy.GEMM:
        # GEMM requires dense matrices: A (nodes x features), B (nodes), C (leaves)
        # Assuming typical float32 arrays
        return (num_nodes * num_nodes * 4) + (num_nodes * 4) + (num_nodes * 4) * batch_size
    elif strategy == Strategy.TREE_TRAVERSAL:
        # Tree traversal uses flat 1D arrays
        return (num_nodes * 6 * 4) * batch_size  # features, thresholds, left, right, values, tracks
    elif strategy == Strategy.PERFECT_TREE_TRAVERSAL:
        # Perfect tree padding can exponentially grow, let's estimate log depth
        import math

        depth = math.ceil(math.log2(num_nodes + 1)) if num_nodes > 0 else 0
        perfect_nodes = (2**depth) - 1
        return (perfect_nodes * 4 * 4) * batch_size
    return 0


def select_optimal_strategy(
    abstractions: TreeAbstractions,
    target: TargetHardware,
    batch_size: int = 1,
    force_strategy: Optional[Strategy] = None,
) -> Strategy:
    """Implement strategy selector based on hardware target, tree depth, sparsity, and batch size."""
    if force_strategy is not None:
        return force_strategy

    num_nodes = len(abstractions.features)
    import math

    depth = math.ceil(math.log2(num_nodes + 1)) if num_nodes > 0 else 0

    # Heuristics based on hardware and batch_size
    if target == TargetHardware.WEBGPU:
        # WebGPU hates branching, prefers GEMM or PerfectTree depending on depth
        if (
            depth > 10
        ):  # Deep tree: PerfectTree memory blows up, use GEMM if sparse, else TreeTraversal
            return Strategy.GEMM  # Or TREE_TRAVERSAL if memory is too big
        else:
            return Strategy.GEMM
    elif target == TargetHardware.GPU:
        if batch_size > 1000:
            return Strategy.GEMM
        return Strategy.GEMM
    else:  # CPU
        if batch_size == 1:
            return Strategy.TREE_TRAVERSAL
        return Strategy.GEMM
