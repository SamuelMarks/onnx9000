from onnx9000.frontends.frontend.nn.functional import interpolate
from onnx9000.frontends.frontend.tensor import Tensor
import numpy as np


def test_interpolate_align_corners_unsupported():
    t = Tensor(np.zeros((1, 1, 2, 2)))
    res = interpolate(
        t, size=(4, 4), scale_factor=None, mode="nearest", align_corners=True
    )
    assert res is None
