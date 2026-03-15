import pytest
from onnx9000.extensions.custom.vision_ops import roi_align


def test_roi_align_out_of_bounds_continue():
    x = [[[[0.0] * 4] * 4]]
    rois = [[-10.0, -10.0, -5.0, -5.0]]
    batch_indices = [0]
    res = roi_align(
        x,
        rois,
        batch_indices,
        output_height=2,
        output_width=2,
        spatial_scale=1.0,
        sampling_ratio=1,
    )
    assert len(res) == 1
    assert len(res[0]) == 1
    assert len(res[0][0]) == 2
    assert len(res[0][0][0]) == 2
