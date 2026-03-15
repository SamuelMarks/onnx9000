"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.extensions.custom.ops import iou, nms, topk, unique


def test_iou():
    """Provides semantic functionality and verification."""
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    assert iou(box1, box2) == 25 / 175
    box3 = [20, 20, 30, 30]
    assert iou(box1, box3) == 0.0


def test_nms():
    """Provides semantic functionality and verification."""
    boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]]
    scores = [0.9, 0.8, 0.95]
    keep = nms(boxes, scores, iou_threshold=0.5, score_threshold=0.85)
    assert keep == [2, 0]
    scores2 = [0.9, 0.95, 0.8]
    keep2 = nms(boxes, scores2, iou_threshold=0.5, score_threshold=0.0)
    assert keep2 == [1, 2]


def test_topk():
    """Provides semantic functionality and verification."""
    arr = [1.0, 5.0, 2.0, 8.0, 3.0]
    vals, idxs = topk(arr, 3)
    assert vals == [8.0, 5.0, 3.0]
    assert idxs == [3, 1, 4]


def test_unique():
    """Provides semantic functionality and verification."""
    arr = [2.0, 1.0, 2.0, 3.0, 1.0]
    vals, indices, inverse, counts = unique(arr)
    assert vals == [1.0, 2.0, 3.0]
    assert indices == [1, 0, 3]
    assert inverse == [1, 0, 1, 2, 0]
    assert counts == [2, 2, 1]
