from onnx9000.extensions.custom.vision_ops import roi_align


def test_vision_ops_cov_final():
    out = roi_align(
        input_tensor=[[[[1.0]]]],
        rois=[[100.0, 100.0, 105.0, 105.0]],
        batch_indices=[0],
        output_height=1,
        output_width=1,
        spatial_scale=1.0,
        sampling_ratio=1,
        aligned=True,
    )
    assert out == [[[[0.0]]]]
