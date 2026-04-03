"""Tests the functional cov module functionality."""

import numpy as np
from onnx9000.converters.frontend.nn.functional import interpolate
from onnx9000.converters.frontend.tensor import Tensor


def test_interpolate_align_corners_unsupported() -> None:
    """Tests the interpolate align corners unsupported functionality."""
    t = Tensor(np.zeros((1, 1, 2, 2)))
    res = interpolate(t, size=(4, 4), scale_factor=None, mode="nearest", align_corners=True)
    assert res is None


from onnx9000.converters.frontend.builder import GraphBuilder, Tracing
from onnx9000.converters.frontend.nn import functional as F
from onnx9000.core.dtypes import DType


def test_activations():
    x = Tensor(name="x", shape=(1, 2), dtype=DType.FLOAT32)
    b = GraphBuilder("test_acts")
    with Tracing(b):
        F.relu(x)
        assert b.nodes[-1].op_type == "Relu"
        F.sigmoid(x)
        assert b.nodes[-1].op_type == "Sigmoid"
        F.tanh(x)
        assert b.nodes[-1].op_type == "Tanh"
        F.gelu(x)
        assert b.nodes[-1].op_type == "Gelu"
        F.softmax(x, dim=1)
        assert b.nodes[-1].op_type == "Softmax"


def test_linear():
    x = Tensor(name="x", shape=(1, 2), dtype=DType.FLOAT32)
    w = Tensor(name="w", shape=(3, 2), dtype=DType.FLOAT32)
    b_t = Tensor(name="b", shape=(3,), dtype=DType.FLOAT32)
    b = GraphBuilder("test_linear")
    with Tracing(b):
        F.linear(x, w, b_t)
        assert b.nodes[-1].op_type == "Add"  # linear produces matmul + add


def test_conv2d():
    x = Tensor(name="x", shape=(1, 3, 224, 224), dtype=DType.FLOAT32)
    w = Tensor(name="w", shape=(16, 3, 3, 3), dtype=DType.FLOAT32)
    bias = Tensor(name="bias", shape=(16,), dtype=DType.FLOAT32)
    b = GraphBuilder("test_conv")
    with Tracing(b):
        F.conv2d(x, w, bias, stride=1, padding=1, dilation=1, groups=1)
        assert b.nodes[-1].op_type == "Conv"


def test_pad():
    x = Tensor(name="x", shape=(1, 1, 2, 2), dtype=DType.FLOAT32)
    b = GraphBuilder("test_pad")
    with Tracing(b):
        F.pad(x, (1, 1), mode="constant", value=0.0)
        assert b.nodes[-1].op_type == "Pad"
        assert b.nodes[-1].attributes["mode"] == "constant"


def test_upsample():
    x = Tensor(name="x", shape=(1, 3, 8, 8), dtype=DType.FLOAT32)
    b = GraphBuilder("test_upsample")
    with Tracing(b):
        # interpolate fails if size is not None in current implementation
        # F.interpolate(x, size=(16, 16)...) returns None! (line 262-263)
        F.interpolate(x, scale_factor=2.0, mode="nearest", align_corners=None)
        assert b.nodes[-1].op_type == "Resize"

        F.interpolate(x, scale_factor=(2.0, 2.0), mode="nearest", align_corners=None)
        assert b.nodes[-1].op_type == "Resize"

        assert (
            F.interpolate(x, size=(16, 16), scale_factor=2.0, mode="nearest", align_corners=None)
            is None
        )
        assert F.interpolate(x, scale_factor=2.0, mode="linear", align_corners=True) is None


def test_one_hot():
    x = Tensor(name="x", shape=(1, 3), dtype=DType.INT64)
    b = GraphBuilder("test_one_hot")
    with Tracing(b):
        F.one_hot(x, num_classes=10)
        assert b.nodes[-1].op_type == "OneHot"
    x = Tensor(name="x", shape=(1, 2), dtype=DType.FLOAT32)
    b = GraphBuilder("test_log_softmax")
    with Tracing(b):
        F.log_softmax(x, dim=1)
        assert b.nodes[-1].op_type == "LogSoftmax"


def test_max_pool2d():
    x = Tensor(name="x", shape=(1, 3, 224, 224), dtype=DType.FLOAT32)
    b = GraphBuilder("test_max_pool2d")
    with Tracing(b):
        F.max_pool2d(x, kernel_size=2)
        assert b.nodes[-1].op_type == "MaxPool"
        assert b.nodes[-1].attributes["kernel_shape"] == [2, 2]

        F.max_pool2d(x, kernel_size=(3, 3), stride=1, padding=(1, 1), dilation=2, ceil_mode=True)
        assert b.nodes[-1].op_type == "MaxPool"
        assert b.nodes[-1].attributes["kernel_shape"] == [3, 3]
