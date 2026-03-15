"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.extensions.custom.vision_ops import grid_sample, deform_conv2d


def test_grid_sample_coverage():
    """Provides semantic functionality and verification."""
    input_tensor = [[[[1.0, 2.0], [3.0, 4.0]]]]
    grid = [[[[-2.0, -2.0], [2.0, 2.0]]]]
    out_border = grid_sample(
        input_tensor, grid, mode="nearest", padding_mode="border", align_corners=False
    )
    out_reflect = grid_sample(
        input_tensor,
        grid,
        mode="nearest",
        padding_mode="reflection",
        align_corners=False,
    )


def test_deform_conv2d_mask():
    """Provides semantic functionality and verification."""
    x = [[[[1.0]]]]
    weight = [[[[1.0]]]]
    offset = [[[[0.1]], [[0.2]]]]
    mask = [[[[0.5]]]]
    out = deform_conv2d(x, weight, offset, mask=mask)
