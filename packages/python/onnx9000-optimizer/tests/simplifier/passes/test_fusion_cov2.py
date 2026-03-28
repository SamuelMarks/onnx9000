"""Tests for packages/python/onnx9000-optimizer/tests/simplifier/passes/test_fusion_cov2.py."""

import numpy as np
import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.optimizer.simplifier.passes.fusion import (
    fuse_batchnorm_into_conv,
    fuse_batchnorm_into_gemm,
    map_aten_arange_to_range,
)


def test_gemm_num_consumers_or():
    """Test gemm num consumers or."""
    g1 = Graph("G1_or")
    n_gemm = Node("Gemm", ["X", "W"], ["gemm_out"])
    g1.nodes.append(n_gemm)
    n_bn = Node("BatchNormalization", ["gemm_out", "s", "b", "m", "v"], ["Y"])
    g1.nodes.append(n_bn)
    g1.outputs = ["gemm_out"]
    fuse_batchnorm_into_gemm(g1)


def test_gemm_w_or():
    """Test gemm w or."""
    g1 = Graph("G1_w_or")
    n_gemm = Node("Gemm", ["X", "W"], ["gemm_out"])
    g1.nodes.append(n_gemm)
    n_bn = Node("BatchNormalization", ["gemm_out", "s", "b", "m", "v"], ["Y"])
    g1.nodes.append(n_bn)
    for name in ["s", "b", "m", "v"]:
        t = Tensor(name, (2,), DType.FLOAT32)
        t.data = np.ones(2, dtype=np.float32)
        t.is_initializer = True
        g1.tensors[name] = t
    w = Tensor("W", (2, 2), DType.FLOAT32)
    w.data = "not_ndarray"
    w.is_initializer = True
    g1.tensors["W"] = w
    fuse_batchnorm_into_gemm(g1)


def test_conv_num_consumers_or():
    """Test conv num consumers or."""
    g1 = Graph("G1_conv_or")
    n_conv = Node("Conv", ["X", "W"], ["conv_out"])
    g1.nodes.append(n_conv)
    n_bn = Node("BatchNormalization", ["conv_out", "s", "b", "m", "v"], ["Y"])
    g1.nodes.append(n_bn)
    g1.outputs = [ValueInfo("conv_out", (), DType.FLOAT32)]
    fuse_batchnorm_into_conv(g1)


def test_conv_w_or():
    """Test conv w or."""
    g1 = Graph("G1_conv_w_or")
    n_conv = Node("Conv", ["X", "W"], ["conv_out"])
    g1.nodes.append(n_conv)
    n_bn = Node("BatchNormalization", ["conv_out", "s", "b", "m", "v"], ["Y"])
    g1.nodes.append(n_bn)
    for name in ["s", "b", "m", "v"]:
        t = Tensor(name, (2,), DType.FLOAT32)
        t.data = np.ones(2, dtype=np.float32)
        t.is_initializer = True
        g1.tensors[name] = t
    w = Tensor("W", (2, 2, 1, 1), DType.FLOAT32)
    w.data = "not_ndarray"
    w.is_initializer = True
    g1.tensors["W"] = w
    fuse_batchnorm_into_conv(g1)


def test_conv_b_or():
    """Test conv b or."""
    g1 = Graph("G1_conv_b_or")
    n_conv = Node("Conv", ["X", "W", "B"], ["conv_out"])
    g1.nodes.append(n_conv)
    n_bn = Node("BatchNormalization", ["conv_out", "s", "b", "m", "v"], ["Y"])
    g1.nodes.append(n_bn)
    for name in ["s", "b", "m", "v"]:
        t = Tensor(name, (2,), DType.FLOAT32)
        t.data = np.ones(2, dtype=np.float32)
        t.is_initializer = True
        g1.tensors[name] = t
    w = Tensor("W", (2, 2, 1, 1), DType.FLOAT32)
    w.data = np.ones((2, 2, 1, 1), dtype=np.float32)
    w.is_initializer = True
    g1.tensors["W"] = w
    b = Tensor("B", (2,), DType.FLOAT32)
    b.data = "not_ndarray"
    b.is_initializer = True
    g1.tensors["B"] = b
    fuse_batchnorm_into_conv(g1)


def test_map_aten_arange_inputs():
    """Test map aten arange inputs."""
    g = Graph("AtenInputs")
    n = Node("arange", ["start", "end"], ["Y"], domain="aten")
    g.nodes.append(n)
    changed = map_aten_arange_to_range(g)
    assert not changed
