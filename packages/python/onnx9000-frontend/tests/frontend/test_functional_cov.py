import numpy as np
from onnx9000.frontend.frontend.nn.functional import interpolate
from onnx9000.frontend.frontend.tensor import Tensor


def test_interpolate_align_corners_unsupported() -> None:
    t = Tensor(np.zeros((1, 1, 2, 2)))
    res = interpolate(t, size=(4, 4), scale_factor=None, mode="nearest", align_corners=True)
    assert res is None
