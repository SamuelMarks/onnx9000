"""Test Normalization."""

from onnx9000 import GraphBuilder, Tracing
from onnx9000.frontends.frontend.nn.normalization import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    LayerNorm,
    GroupNorm,
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
)
from onnx9000.frontends.frontend.tensor import Tensor
from onnx9000.core.dtypes import DType


def test_normalization():
    """Provides semantic functionality and verification."""
    bn = BatchNorm2d(16)
    ln = LayerNorm((16, 50, 50))
    gn = GroupNorm(4, 16)
    in_ = InstanceNorm2d(16)
    assert bn.weight.shape == (16,)
    assert ln.weight.shape == (16, 50, 50)
    assert gn.weight.shape == (16,)
    assert in_.weight is None
    builder = GraphBuilder("dummy")
    with Tracing(builder):
        x = Tensor((20, 16, 50, 50), DType.FLOAT32, "x")
        y1 = bn(x)
        y2 = ln(x)
        y3 = gn(x)
        y4 = in_(x)
        bn_na = BatchNorm1d(16, affine=False, track_running_stats=False)
        x_1d = Tensor((20, 16, 50), DType.FLOAT32, "x1d")
        y5 = bn_na(x_1d)
    assert y1 is not None
    assert y2 is not None
    assert y3 is not None
    assert y4 is not None
    assert y5 is not None
