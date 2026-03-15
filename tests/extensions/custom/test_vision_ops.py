"""Module providing core logic and structural definitions."""

import pytest
import math
from onnx9000.extensions.custom.vision_ops import (
    rotated_nms,
    grid_sample,
    roi_align,
    deform_conv2d,
)


def test_rotated_nms():
    """Provides semantic functionality and verification."""
    boxes = [[5, 5, 10, 10, 0.0], [5, 5, 10, 10, math.pi / 2], [20, 20, 10, 10, 0.0]]
    scores = [0.9, 0.8, 0.95]
    keep = rotated_nms(boxes, scores, iou_threshold=0.5)
    assert keep == [2, 0]


def test_grid_sample():
    """Provides semantic functionality and verification."""
    input_t = [[[[1.0, 2.0], [3.0, 4.0]]]]
    grid = [[[[0.0, 0.0]]]]
    out = grid_sample(input_t, grid, align_corners=True)
    assert math.isclose(out[0][0][0][0], 2.5, abs_tol=1e-05)


def test_roi_align():
    """Provides semantic functionality and verification."""
    input_t = [[[[1.0, 2.0], [3.0, 4.0]]]]
    rois = [[0.0, 0.0, 1.0, 1.0]]
    batch_indices = [0]
    out = roi_align(
        input_t,
        rois,
        batch_indices,
        output_height=2,
        output_width=2,
        spatial_scale=1.0,
        sampling_ratio=1,
        aligned=False,
    )
    assert len(out) == 1
    assert len(out[0][0]) == 2


def test_deform_conv2d():
    """Provides semantic functionality and verification."""
    x = [[[[1.0]]]]
    weight = [[[[1.0]]]]
    offset = [[[[0.0]], [[0.0]]]]
    out = deform_conv2d(x, weight, offset, stride=1, padding=0, dilation=1)
    assert out[0][0][0][0] == 1.0
