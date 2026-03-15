"""Module providing core logic and structural definitions."""

from typing import List, Tuple
import math


def iou(box1: List[float], box2: List[float]) -> float:
    """Provides semantic functionality and verification."""
    # box: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter_area / (area1 + area2 - inter_area)


def nms(
    boxes: List[List[float]],
    scores: List[float],
    iou_threshold: float,
    score_threshold: float = 0.0,
) -> List[int]:
    """
    Non-Maximum Suppression native implementation.
    """
    indices = [i for i, s in enumerate(scores) if s >= score_threshold]
    indices.sort(key=lambda i: scores[i], reverse=True)

    keep = []
    while indices:
        curr = indices.pop(0)
        keep.append(curr)

        indices = [i for i in indices if iou(boxes[curr], boxes[i]) <= iou_threshold]

    return keep


def topk(arr: List[float], k: int) -> Tuple[List[float], List[int]]:
    """
    Returns (values, indices) of top k elements.
    """
    indexed = [(val, i) for i, val in enumerate(arr)]
    indexed.sort(key=lambda x: x[0], reverse=True)

    top = indexed[:k]
    return [x[0] for x in top], [x[1] for x in top]


def unique(arr: List[float]) -> Tuple[List[float], List[int], List[int], List[int]]:
    """
    Implements ONNX Unique operator natively.
    Returns:
    - unique values
    - indices of the first occurrences
    - inverse indices (reconstructs original array)
    - counts of each unique value
    """
    seen = {}
    vals = []
    indices = []
    counts = []

    inverse_indices = []

    for i, val in enumerate(arr):
        if val not in seen:
            seen[val] = len(vals)
            vals.append(val)
            indices.append(i)
            counts.append(1)
        else:
            counts[seen[val]] += 1

        inverse_indices.append(seen[val])

    # ONNX requires values to be sorted in ascending order if specified,
    # but the default behavior is often sorted. Let's return them in sorted order.
    # We will map them to the original indices to get old_to_new mapping correctly.
    # "seen" maps value to the index in `vals` array (which is unsorted).
    sorted_pairs = sorted(
        [
            (v, i, c, old_idx)
            for old_idx, (v, i, c) in enumerate(zip(vals, indices, counts))
        ]
    )

    sorted_vals = [x[0] for x in sorted_pairs]
    sorted_indices = [x[1] for x in sorted_pairs]
    sorted_counts = [x[2] for x in sorted_pairs]

    # We must remap inverse_indices to the sorted mapping
    # old_idx is the index in the original `vals` array
    old_to_new = {x[3]: new_idx for new_idx, x in enumerate(sorted_pairs)}

    new_inverse = [old_to_new[seen[val]] for val in arr]

    return sorted_vals, sorted_indices, new_inverse, sorted_counts
