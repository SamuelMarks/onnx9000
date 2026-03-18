import pytest
import numpy as np
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.core.dtypes import DType
from onnx9000.optimizer.simplifier.passes.fusion import (
    fuse_batchnorm_into_gemm,
    fuse_batchnorm_into_conv,
    map_aten_arange_to_range,
)


def test_gemm_num_consumers_or():
    # 280: bn_x in graph.outputs
    g1 = Graph("G1_or")
    n_gemm = Node("Gemm", ["X", "W"], ["gemm_out"])
    g1.nodes.append(n_gemm)
    n_bn = Node("BatchNormalization", ["gemm_out", "s", "b", "m", "v"], ["Y"])
    g1.nodes.append(n_bn)
    g1.outputs = ["gemm_out"]  # num_consumers=1 but bn_x is output
    fuse_batchnorm_into_gemm(g1)


def test_gemm_w_or():
    # 325: gemm_w_t exists, is_init=True, but not ndarray
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
    # 415: bn_x in graph.outputs
    g1 = Graph("G1_conv_or")
    n_conv = Node("Conv", ["X", "W"], ["conv_out"])
    g1.nodes.append(n_conv)
    n_bn = Node("BatchNormalization", ["conv_out", "s", "b", "m", "v"], ["Y"])
    g1.nodes.append(n_bn)
    g1.outputs = [ValueInfo("conv_out", (), DType.FLOAT32)]
    fuse_batchnorm_into_conv(g1)


def test_conv_w_or():
    # 449: conv_w_t exists, is_init=True, but not ndarray
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
    # 461: conv_b_t exists, is_init=True, but not ndarray
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
    # 561: len(node.inputs) < 3
    g = Graph("AtenInputs")
    n = Node("arange", ["start", "end"], ["Y"], domain="aten")
    g.nodes.append(n)
    changed = map_aten_arange_to_range(g)
    assert not changed
