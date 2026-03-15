"""Test init and functional."""

from onnx9000.frontends.frontend.nn import init
from onnx9000.frontends.frontend.nn import functional as F
from onnx9000.frontends.frontend.tensor import Tensor
from onnx9000.core.dtypes import DType
import numpy as np
import pytest


def test_init():
    """Provides semantic functionality and verification."""
    t = Tensor((10, 20), DType.FLOAT32, data=np.zeros((10, 20), dtype=np.float32))
    init.xavier_uniform_(t)
    assert not np.allclose(t.data, 0.0)
    init.zeros_(t)
    assert np.allclose(t.data, 0.0)
    init.ones_(t)
    assert np.allclose(t.data, 1.0)
    init.constant_(t, 5.0)
    assert np.allclose(t.data, 5.0)
    init.xavier_normal_(t)
    init.kaiming_uniform_(t)
    init.kaiming_normal_(t)
    t_1d = Tensor((10,), DType.FLOAT32)
    with pytest.raises(ValueError):
        init.calculate_fan_in_and_fan_out(t_1d)


def test_functional():
    """Provides semantic functionality and verification."""
    from onnx9000 import GraphBuilder, Tracing

    builder = GraphBuilder("dummy")
    with Tracing(builder):
        x = Tensor((2, 16, 50, 50), DType.FLOAT32, "x")
        w = Tensor((33, 16, 3, 3), DType.FLOAT32, "w")
        y1 = F.relu(x)
        y2 = F.sigmoid(x)
        y3 = F.tanh(x)
        y4 = F.gelu(x)
        y5 = F.softmax(x)
        y6 = F.conv2d(x, w)
        y7 = F.pad(x, (1, 1, 2, 2))
        x2 = Tensor((2, 10), DType.FLOAT32, "x2")
        w2 = Tensor((5, 10), DType.FLOAT32, "w2")
        y8 = F.linear(x2, w2)
        y9 = F.interpolate(x, scale_factor=2.0)
        idx = Tensor((10,), DType.INT64, "idx")
        y10 = F.one_hot(idx, 5)
    assert y1 is not None
