import logging

from onnx9000.optimizer.hummingbird.memory import TreeAbstractions

logger = logging.getLogger(__name__)


def analyze_tree_depth(abstractions: TreeAbstractions) -> dict[str, float]:
    """Provide tree depth analysis utility (min, max, mean depths)."""
    if not abstractions.features:
        return {"min": 0, "max": 0, "mean": 0}

    depths = []

    # We trace paths from roots
    def trace(node_idx: int, current_depth: int) -> None:
        if (
            abstractions.left_children[node_idx] == -1
            and abstractions.right_children[node_idx] == -1
        ):
            depths.append(current_depth)
            return
        if abstractions.left_children[node_idx] != -1:
            trace(abstractions.left_children[node_idx], current_depth + 1)
        if abstractions.right_children[node_idx] != -1:
            trace(abstractions.right_children[node_idx], current_depth + 1)

    # Assuming node 0 is the root for a single tree representation
    trace(0, 1)

    if not depths:
        return {"min": 0, "max": 0, "mean": 0}

    return {"min": min(depths), "max": max(depths), "mean": sum(depths) / len(depths)}


def analyze_leaf_distribution(abstractions: TreeAbstractions) -> dict[float, int]:
    """Provide tree leaf distribution utility."""
    distribution = {}
    for i, left in enumerate(abstractions.left_children):
        if left == -1:  # It's a leaf
            val = abstractions.values[i]
            distribution[val] = distribution.get(val, 0) + 1
    return distribution


def flatten_ensemble(trees: list[TreeAbstractions]) -> TreeAbstractions:
    """Flatten nested ensemble structures into unified 2D/3D tensors."""
    flattened = TreeAbstractions()
    offset = 0
    for tree in trees:
        for i in range(len(tree.features)):
            left = tree.left_children[i] + offset if tree.left_children[i] != -1 else -1
            right = tree.right_children[i] + offset if tree.right_children[i] != -1 else -1
            flattened.add_node(
                feature=tree.features[i],
                threshold=tree.thresholds[i],
                left=left,
                right=right,
                value=tree.values[i],
                missing=tree.missing_tracks[i] if hasattr(tree, "missing_tracks") else 0,
            )
        offset += len(tree.features)
    return flattened


def cast_parameters(abstractions: TreeAbstractions, target_dtype="float32") -> TreeAbstractions:
    """Resolve numerical precision mismatches (FP32 vs FP64) ahead of time.
    Cast FP64 parameters to FP32 natively to optimize WebGPU limits.
    """
    casted = TreeAbstractions()
    for i in range(len(abstractions.features)):
        # Simulate casting Python floats (which are double/FP64) to FP32 limits
        import struct

        val32 = struct.unpack("f", struct.pack("f", abstractions.values[i]))[0]
        thresh32 = struct.unpack("f", struct.pack("f", abstractions.thresholds[i]))[0]

        casted.add_node(
            feature=abstractions.features[i],
            threshold=thresh32,
            left=abstractions.left_children[i],
            right=abstractions.right_children[i],
            value=val32,
            missing=abstractions.missing_tracks[i],
        )
    return casted
