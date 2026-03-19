import numpy as np
import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor, ValueInfo
from onnx9000.optimizer.simplifier.passes.fusion import (
    fuse_batchnorm_into_conv,
    fuse_batchnorm_into_gemm,
    map_aten_arange_to_range,
)


def test_gemm_out_name_value_info():
    # 374: out.name = bn_x in gemm fusion
    g4 = Graph("G4_exact")
    n_gemm4 = Node("Gemm", ["X", "W"], ["gemm_out"])
    g4.nodes.append(n_gemm4)
    g4.nodes.append(Node("BatchNormalization", ["gemm_out", "s", "b", "m", "v"], ["Y"]))

    for name in ["s", "b", "m", "v"]:
        t = Tensor(name, (2,), DType.FLOAT32)
        t.data = np.ones(2, dtype=np.float32)
        t.is_initializer = True
        g4.tensors[name] = t
        g4.initializers.append(name)

    w = Tensor("W", (2, 2), DType.FLOAT32)
    w.data = np.eye(2, dtype=np.float32)
    w.is_initializer = True
    g4.tensors["W"] = w
    g4.initializers.append("W")

    v = ValueInfo("Y", (), DType.FLOAT32)
    g4.outputs = [v]

    changed = fuse_batchnorm_into_gemm(g4)
    assert changed
    assert v.name == "gemm_out"


def test_conv_out_name_value_info():
    # 506: out.name = bn_x in conv fusion
    g4 = Graph("G4_exact_conv")
    n_conv4 = Node("Conv", ["X", "W"], ["conv_out"])
    g4.nodes.append(n_conv4)
    g4.nodes.append(Node("BatchNormalization", ["conv_out", "s", "b", "m", "v"], ["Y"]))

    for name in ["s", "b", "m", "v"]:
        t = Tensor(name, (2,), DType.FLOAT32)
        t.data = np.ones(2, dtype=np.float32)
        t.is_initializer = True
        g4.tensors[name] = t
        g4.initializers.append(name)

    w = Tensor("W", (2, 2, 1, 1), DType.FLOAT32)
    w.data = np.ones((2, 2, 1, 1), dtype=np.float32)
    w.is_initializer = True
    g4.tensors["W"] = w
    g4.initializers.append("W")

    v = ValueInfo("Y", (), DType.FLOAT32)
    g4.outputs = [v]

    changed = fuse_batchnorm_into_conv(g4)
    assert changed
    assert v.name == "conv_out"


def test_aten_arange_missing_inputs():
    # 561: len(node.inputs) < 3, so not changed
    g = Graph("ATenMissing")
    n = Node("arange", ["start", "end"], ["Y"], domain="aten")
    g.nodes.append(n)
    changed = map_aten_arange_to_range(g)
    assert not changed


def test_gemm_out_name_value_info_consumer():
    g4 = Graph("G4_cons")
    n_gemm4 = Node("Gemm", ["X", "W"], ["gemm_out"])
    g4.nodes.append(n_gemm4)
    g4.nodes.append(Node("BatchNormalization", ["gemm_out", "s", "b", "m", "v"], ["Y"]))
    g4.nodes.append(Node("Abs", ["Y"], ["Z"]))  # 374: Consumer of BN output!

    for name in ["s", "b", "m", "v"]:
        t = Tensor(name, (2,), DType.FLOAT32)
        t.data = np.ones(2, dtype=np.float32)
        t.is_initializer = True
        g4.tensors[name] = t
        g4.initializers.append(name)

    w = Tensor("W", (2, 2), DType.FLOAT32)
    w.data = np.eye(2, dtype=np.float32)
    w.is_initializer = True
    g4.tensors["W"] = w
    g4.initializers.append("W")

    fuse_batchnorm_into_gemm(g4)


def test_conv_out_name_value_info_consumer():
    g4 = Graph("G4_exact_conv_cons")
    n_conv4 = Node("Conv", ["X", "W"], ["conv_out"])
    g4.nodes.append(n_conv4)
    g4.nodes.append(Node("BatchNormalization", ["conv_out", "s", "b", "m", "v"], ["Y"]))
    g4.nodes.append(Node("Abs", ["Y"], ["Z"]))  # 506: Consumer of BN output!

    for name in ["s", "b", "m", "v"]:
        t = Tensor(name, (2,), DType.FLOAT32)
        t.data = np.ones(2, dtype=np.float32)
        t.is_initializer = True
        g4.tensors[name] = t
        g4.initializers.append(name)

    w = Tensor("W", (2, 2, 1, 1), DType.FLOAT32)
    w.data = np.ones((2, 2, 1, 1), dtype=np.float32)
    w.is_initializer = True
    g4.tensors["W"] = w
    g4.initializers.append("W")

    fuse_batchnorm_into_conv(g4)


def test_run_all_fusions_pass():
    from onnx9000.optimizer.simplifier.passes.fusion import run_all_fusions

    g = Graph("G_all_pass")
    n = Node("arange", ["start", "end", "step"], ["Y"], domain="aten")
    g.nodes.append(n)

    # map_aten_arange_to_range will return True
    run_all_fusions(g)  # 561
