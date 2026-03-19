import numpy as np
import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor, ValueInfo
from onnx9000.optimizer.simplifier.passes.fusion import (
    fuse_batchnorm_into_conv,
    fuse_batchnorm_into_gemm,
    fuse_consecutive_transpose,
    fuse_linear_activation,
    fuse_matmul_add,
    map_aten_arange_to_range,
)


def test_fuse_batchnorm_into_gemm():
    g = Graph("TestBNGemm")
    g.inputs = ["X"]
    g.outputs = ["Y"]
    g.tensors["X"] = Tensor("X", (2, 2), DType.FLOAT32)

    w = Tensor("W", (2, 2), DType.FLOAT32)
    w.data = np.eye(2, dtype=np.float32)
    w.is_initializer = True
    g.tensors["W"] = w

    b_gemm = Tensor("B_gemm", (2,), DType.FLOAT32)
    b_gemm.data = np.zeros(2, dtype=np.float32)
    b_gemm.is_initializer = True
    g.tensors["B_gemm"] = b_gemm

    # Gemm out
    g.tensors["gemm_out"] = Tensor("gemm_out", (2, 2), DType.FLOAT32)

    n_gemm = Node("Gemm", ["X", "W", "B_gemm"], ["gemm_out"])
    n_gemm.attributes["transB"] = Attribute("transB", "INT", 1)  # Must be 1
    g.nodes.append(n_gemm)

    scale = Tensor("scale", (2,), DType.FLOAT32)
    scale.data = np.ones(2, dtype=np.float32)
    scale.is_initializer = True
    g.tensors["scale"] = scale

    b_bn = Tensor("B_bn", (2,), DType.FLOAT32)
    b_bn.data = np.ones(2, dtype=np.float32)
    b_bn.is_initializer = True
    g.tensors["B_bn"] = b_bn

    mean = Tensor("mean", (2,), DType.FLOAT32)
    mean.data = np.zeros(2, dtype=np.float32)
    mean.is_initializer = True
    g.tensors["mean"] = mean

    var = Tensor("var", (2,), DType.FLOAT32)
    var.data = np.ones(2, dtype=np.float32)
    var.is_initializer = True
    g.tensors["var"] = var

    g.tensors["Y"] = Tensor("Y", (2, 2), DType.FLOAT32)

    n_bn = Node("BatchNormalization", ["gemm_out", "scale", "B_bn", "mean", "var"], ["Y"])
    g.nodes.append(n_bn)

    # Needs to be initializers list
    for name in ["W", "B_gemm", "scale", "B_bn", "mean", "var"]:
        g.initializers.append(name)

    changed = fuse_batchnorm_into_gemm(g)
    assert changed

    # Should skip if transB is 0
    g2 = Graph("TestBNGemm2")
    w2 = Tensor("W2", (2, 2), DType.FLOAT32)
    w2.data = np.eye(2, dtype=np.float32)
    w2.is_initializer = True
    g2.tensors["W2"] = w2

    n_gemm2 = Node("Gemm", ["X", "W2"], ["gemm_out"])
    g2.nodes.append(n_gemm2)

    n_bn2 = Node("BatchNormalization", ["gemm_out", "scale", "B_bn", "mean", "var"], ["Y"])
    g2.nodes.append(n_bn2)

    for name in ["W2", "scale", "B_bn", "mean", "var"]:
        g2.initializers.append(name)
        if name not in g2.tensors:
            g2.tensors[name] = g.tensors[name]

    changed2 = fuse_batchnorm_into_gemm(g2)
    assert changed2  # transB 0 handled correctly


def test_fuse_batchnorm_into_conv():
    g = Graph("TestBNConv")
    g.inputs = ["X"]
    g.outputs = ["Y"]
    g.tensors["X"] = Tensor("X", (1, 2, 2, 2), DType.FLOAT32)

    w = Tensor("W", (2, 2, 1, 1), DType.FLOAT32)
    w.data = np.ones((2, 2, 1, 1), dtype=np.float32)
    w.is_initializer = True
    g.tensors["W"] = w

    b_conv = Tensor("B_conv", (2,), DType.FLOAT32)
    b_conv.data = np.zeros(2, dtype=np.float32)
    b_conv.is_initializer = True
    g.tensors["B_conv"] = b_conv

    g.tensors["conv_out"] = Tensor("conv_out", (1, 2, 2, 2), DType.FLOAT32)

    n_conv = Node("Conv", ["X", "W", "B_conv"], ["conv_out"])
    g.nodes.append(n_conv)

    scale = Tensor("scale", (2,), DType.FLOAT32)
    scale.data = np.ones(2, dtype=np.float32)
    scale.is_initializer = True
    g.tensors["scale"] = scale

    b_bn = Tensor("B_bn", (2,), DType.FLOAT32)
    b_bn.data = np.ones(2, dtype=np.float32)
    b_bn.is_initializer = True
    g.tensors["B_bn"] = b_bn

    mean = Tensor("mean", (2,), DType.FLOAT32)
    mean.data = np.zeros(2, dtype=np.float32)
    mean.is_initializer = True
    g.tensors["mean"] = mean

    var = Tensor("var", (2,), DType.FLOAT32)
    var.data = np.ones(2, dtype=np.float32)
    var.is_initializer = True
    g.tensors["var"] = var

    g.tensors["Y"] = Tensor("Y", (1, 2, 2, 2), DType.FLOAT32)

    n_bn = Node("BatchNormalization", ["conv_out", "scale", "B_bn", "mean", "var"], ["Y"])
    g.nodes.append(n_bn)

    for name in ["W", "B_conv", "scale", "B_bn", "mean", "var"]:
        g.initializers.append(name)

    changed = fuse_batchnorm_into_conv(g)
    assert changed


def test_map_aten_arange_to_range():
    g = Graph("TestAtenArange")
    n = Node("arange", ["start", "end", "step", "dtype"], ["Y"], domain="aten")
    n.attributes["operator"] = Attribute("operator", "STRING", b"arange")
    g.nodes.append(n)

    changed = map_aten_arange_to_range(g)
    assert changed
    assert g.nodes[0].op_type == "Range"
    assert g.nodes[0].domain == ""


def test_fuse_batchnorm_into_gemm_missing_inputs():
    g = Graph("TestBNGemmMissing")
    g.inputs = ["X"]
    g.outputs = ["Y"]
    g.tensors["X"] = Tensor("X", (2, 2), DType.FLOAT32)

    # 1. Gemm missing W
    n_gemm = Node("Gemm", ["X", "W"], ["gemm_out"])
    g.nodes.append(n_gemm)

    n_bn = Node("BatchNormalization", ["gemm_out", "scale", "B_bn", "mean", "var"], ["Y"])
    g.nodes.append(n_bn)
    changed = fuse_batchnorm_into_gemm(g)
    assert not changed

    # 2. BN missing inputs
    w = Tensor("W", (2, 2), DType.FLOAT32)
    w.data = np.eye(2, dtype=np.float32)
    w.is_initializer = True
    g.tensors["W"] = w
    g.initializers.append("W")

    changed = fuse_batchnorm_into_gemm(g)
    assert not changed

    # 3. BN inputs not initializers
    scale = Tensor("scale", (2,), DType.FLOAT32)
    scale.data = np.ones(2, dtype=np.float32)  # not init
    g.tensors["scale"] = scale
    b_bn = Tensor("B_bn", (2,), DType.FLOAT32)
    b_bn.data = np.ones(2, dtype=np.float32)
    g.tensors["B_bn"] = b_bn
    mean = Tensor("mean", (2,), DType.FLOAT32)
    mean.data = np.zeros(2, dtype=np.float32)
    g.tensors["mean"] = mean
    var = Tensor("var", (2,), DType.FLOAT32)
    var.data = np.ones(2, dtype=np.float32)
    g.tensors["var"] = var
    changed = fuse_batchnorm_into_gemm(g)
    assert not changed

    # 4. Gemm W not ndarray
    scale.is_initializer = True
    b_bn.is_initializer = True
    mean.is_initializer = True
    var.is_initializer = True
    w.data = 1.0  # not ndarray
    g.tensors["scale"] = scale
    g.tensors["B_bn"] = b_bn
    g.tensors["mean"] = mean
    g.tensors["var"] = var
    changed = fuse_batchnorm_into_gemm(g)
    assert not changed

    # 5. Gemm B not initializer
    w.data = np.eye(2, dtype=np.float32)
    b_gemm = Tensor("B_gemm", (2,), DType.FLOAT32)
    b_gemm.data = np.zeros(2, dtype=np.float32)  # not init
    g.tensors["B_gemm"] = b_gemm
    n_gemm.inputs.append("B_gemm")
    changed = fuse_batchnorm_into_gemm(g)
    assert not changed

    # 6. BN multiple consumers
    n_other = Node("Abs", ["gemm_out"], ["other_out"])
    g.nodes.append(n_other)
    changed = fuse_batchnorm_into_gemm(g)
    assert not changed


def test_fuse_batchnorm_into_conv_missing_inputs():
    g = Graph("TestBNConvMissing")
    g.inputs = ["X"]
    g.outputs = ["Y"]
    g.tensors["X"] = Tensor("X", (1, 2, 2, 2), DType.FLOAT32)

    # 1. Conv missing W
    n_conv = Node("Conv", ["X", "W"], ["conv_out"])
    g.nodes.append(n_conv)
    n_bn = Node("BatchNormalization", ["conv_out", "scale", "B_bn", "mean", "var"], ["Y"])
    g.nodes.append(n_bn)
    changed = fuse_batchnorm_into_conv(g)
    assert not changed

    # 2. W not init
    w = Tensor("W", (2, 2, 1, 1), DType.FLOAT32)
    w.data = np.ones((2, 2, 1, 1), dtype=np.float32)
    g.tensors["W"] = w
    changed = fuse_batchnorm_into_conv(g)
    assert not changed

    # 3. B not init
    w.is_initializer = True
    g.initializers.append("W")

    scale = Tensor("scale", (2,), DType.FLOAT32)
    scale.data = np.ones(2, dtype=np.float32)
    scale.is_initializer = True
    g.tensors["scale"] = scale
    g.initializers.append("scale")

    b_bn = Tensor("B_bn", (2,), DType.FLOAT32)
    b_bn.data = np.ones(2, dtype=np.float32)
    b_bn.is_initializer = True
    g.tensors["B_bn"] = b_bn
    g.initializers.append("B_bn")

    mean = Tensor("mean", (2,), DType.FLOAT32)
    mean.data = np.zeros(2, dtype=np.float32)
    mean.is_initializer = True
    g.tensors["mean"] = mean
    g.initializers.append("mean")

    var = Tensor("var", (2,), DType.FLOAT32)
    var.data = np.ones(2, dtype=np.float32)
    var.is_initializer = True
    g.tensors["var"] = var
    g.initializers.append("var")

    b_conv = Tensor("B_conv", (2,), DType.FLOAT32)
    b_conv.data = np.zeros(2, dtype=np.float32)  # not init
    g.tensors["B_conv"] = b_conv
    n_conv.inputs.append("B_conv")

    changed = fuse_batchnorm_into_conv(g)
    assert not changed


def test_fusion_gemm_missing_lines():
    # 280: No gemm
    g1 = Graph("G1")
    g1.tensors["scale"] = Tensor("scale", (2,), DType.FLOAT32)
    g1.tensors["scale"].data = np.ones(2, dtype=np.float32)
    g1.tensors["scale"].is_initializer = True
    g1.initializers.append("scale")

    g1.tensors["b"] = Tensor("b", (2,), DType.FLOAT32)
    g1.tensors["b"].data = np.ones(2, dtype=np.float32)
    g1.tensors["b"].is_initializer = True
    g1.initializers.append("b")

    g1.tensors["m"] = Tensor("m", (2,), DType.FLOAT32)
    g1.tensors["m"].data = np.zeros(2, dtype=np.float32)
    g1.tensors["m"].is_initializer = True
    g1.initializers.append("m")

    g1.tensors["v"] = Tensor("v", (2,), DType.FLOAT32)
    g1.tensors["v"].data = np.ones(2, dtype=np.float32)
    g1.tensors["v"].is_initializer = True
    g1.initializers.append("v")

    n_bn = Node("BatchNormalization", ["NOT_GEMM", "scale", "b", "m", "v"], ["Y"])
    g1.nodes.append(n_bn)
    fuse_batchnorm_into_gemm(g1)  # 280
    fuse_batchnorm_into_conv(g1)  # 415

    # 300: Not all ndarray
    g2 = Graph("G2")
    n_gemm = Node("Gemm", ["X", "W"], ["gemm_out"])
    g2.nodes.append(n_gemm)

    g2.tensors["scale"] = Tensor("scale", (2,), DType.FLOAT32)
    g2.tensors["scale"].data = 1.0  # not ndarray
    g2.tensors["scale"].is_initializer = True
    g2.initializers.append("scale")

    g2.tensors["b"] = Tensor("b", (2,), DType.FLOAT32)
    g2.tensors["b"].data = np.ones(2, dtype=np.float32)
    g2.tensors["b"].is_initializer = True
    g2.initializers.append("b")

    g2.tensors["m"] = Tensor("m", (2,), DType.FLOAT32)
    g2.tensors["m"].data = np.zeros(2, dtype=np.float32)
    g2.tensors["m"].is_initializer = True
    g2.initializers.append("m")

    g2.tensors["v"] = Tensor("v", (2,), DType.FLOAT32)
    g2.tensors["v"].data = np.ones(2, dtype=np.float32)
    g2.tensors["v"].is_initializer = True
    g2.initializers.append("v")

    n_bn2 = Node("BatchNormalization", ["gemm_out", "scale", "b", "m", "v"], ["Y"])
    g2.nodes.append(n_bn2)
    fuse_batchnorm_into_gemm(g2)  # 300

    n_conv = Node("Conv", ["X", "W"], ["gemm_out"])
    g2.nodes[0] = n_conv
    fuse_batchnorm_into_conv(g2)  # 436

    # 325: gemm_b not valid
    g3 = Graph("G3")
    n_gemm3 = Node("Gemm", ["X", "W", "B"], ["gemm_out"])
    g3.nodes.append(n_gemm3)

    g3.tensors["scale"] = Tensor("scale", (2,), DType.FLOAT32)
    g3.tensors["scale"].data = np.ones(2, dtype=np.float32)
    g3.tensors["scale"].is_initializer = True
    g3.initializers.append("scale")

    g3.tensors["b"] = Tensor("b", (2,), DType.FLOAT32)
    g3.tensors["b"].data = np.ones(2, dtype=np.float32)
    g3.tensors["b"].is_initializer = True
    g3.initializers.append("b")

    g3.tensors["m"] = Tensor("m", (2,), DType.FLOAT32)
    g3.tensors["m"].data = np.zeros(2, dtype=np.float32)
    g3.tensors["m"].is_initializer = True
    g3.initializers.append("m")

    g3.tensors["v"] = Tensor("v", (2,), DType.FLOAT32)
    g3.tensors["v"].data = np.ones(2, dtype=np.float32)
    g3.tensors["v"].is_initializer = True
    g3.initializers.append("v")

    g3.tensors["W"] = Tensor("W", (2, 2), DType.FLOAT32)
    g3.tensors["W"].data = np.eye(2, dtype=np.float32)
    g3.tensors["W"].is_initializer = True
    g3.initializers.append("W")

    # gemm b missing tensor
    n_bn3 = Node("BatchNormalization", ["gemm_out", "scale", "b", "m", "v"], ["Y"])
    g3.nodes.append(n_bn3)
    fuse_batchnorm_into_gemm(g3)  # 325

    n_conv3 = Node("Conv", ["X", "W", "B"], ["gemm_out"])
    g3.nodes[0] = n_conv3
    fuse_batchnorm_into_conv(g3)  # 461

    # 374: output renaming string
    g4 = Graph("G4")
    g4.outputs = ["Y"]  # String output

    n_gemm4 = Node("Gemm", ["X", "W"], ["gemm_out"])
    g4.nodes.append(n_gemm4)

    g4.tensors["scale"] = Tensor("scale", (2,), DType.FLOAT32)
    g4.tensors["scale"].data = np.ones(2, dtype=np.float32)
    g4.tensors["scale"].is_initializer = True
    g4.initializers.append("scale")

    g4.tensors["b"] = Tensor("b", (2,), DType.FLOAT32)
    g4.tensors["b"].data = np.ones(2, dtype=np.float32)
    g4.tensors["b"].is_initializer = True
    g4.initializers.append("b")

    g4.tensors["m"] = Tensor("m", (2,), DType.FLOAT32)
    g4.tensors["m"].data = np.zeros(2, dtype=np.float32)
    g4.tensors["m"].is_initializer = True
    g4.initializers.append("m")

    g4.tensors["v"] = Tensor("v", (2,), DType.FLOAT32)
    g4.tensors["v"].data = np.ones(2, dtype=np.float32)
    g4.tensors["v"].is_initializer = True
    g4.initializers.append("v")

    g4.tensors["W"] = Tensor("W", (2, 2), DType.FLOAT32)
    g4.tensors["W"].data = np.eye(2, dtype=np.float32)
    g4.tensors["W"].is_initializer = True
    g4.initializers.append("W")

    n_bn4 = Node("BatchNormalization", ["gemm_out", "scale", "b", "m", "v"], ["Y"])
    g4.nodes.append(n_bn4)
    fuse_batchnorm_into_gemm(g4)  # 374
    assert g4.outputs[0] == "gemm_out"


def test_fusion_conv_string_output():
    g4 = Graph("G4C")
    g4.outputs = ["Y"]  # String output

    n_gemm4 = Node("Conv", ["X", "W"], ["gemm_out"])
    g4.nodes.append(n_gemm4)

    g4.tensors["scale"] = Tensor("scale", (2,), DType.FLOAT32)
    g4.tensors["scale"].data = np.ones(2, dtype=np.float32)
    g4.tensors["scale"].is_initializer = True
    g4.initializers.append("scale")

    g4.tensors["b"] = Tensor("b", (2,), DType.FLOAT32)
    g4.tensors["b"].data = np.ones(2, dtype=np.float32)
    g4.tensors["b"].is_initializer = True
    g4.initializers.append("b")

    g4.tensors["m"] = Tensor("m", (2,), DType.FLOAT32)
    g4.tensors["m"].data = np.zeros(2, dtype=np.float32)
    g4.tensors["m"].is_initializer = True
    g4.initializers.append("m")

    g4.tensors["v"] = Tensor("v", (2,), DType.FLOAT32)
    g4.tensors["v"].data = np.ones(2, dtype=np.float32)
    g4.tensors["v"].is_initializer = True
    g4.initializers.append("v")

    g4.tensors["W"] = Tensor("W", (2, 2, 1, 1), DType.FLOAT32)
    g4.tensors["W"].data = np.ones((2, 2, 1, 1), dtype=np.float32)
    g4.tensors["W"].is_initializer = True
    g4.initializers.append("W")

    n_bn4 = Node("BatchNormalization", ["gemm_out", "scale", "b", "m", "v"], ["Y"])
    g4.nodes.append(n_bn4)
    fuse_batchnorm_into_conv(g4)  # 506
    assert g4.outputs[0] == "gemm_out"


def test_fusion_gemm_missing_lines_exact():
    # 280: num_consumers > 1
    g1 = Graph("G1")
    n_gemm = Node("Gemm", ["X", "W"], ["gemm_out"])
    g1.nodes.append(n_gemm)
    g1.nodes.append(Node("BatchNormalization", ["gemm_out", "s", "b", "m", "v"], ["Y"]))
    g1.nodes.append(Node("Abs", ["gemm_out"], ["Z"]))  # second consumer
    fuse_batchnorm_into_gemm(g1)  # 280

    # 300: not all ndarray
    g2 = Graph("G2")
    n_gemm2 = Node("Gemm", ["X", "W"], ["gemm_out"])
    g2.nodes.append(n_gemm2)
    g2.nodes.append(Node("BatchNormalization", ["gemm_out", "s", "b", "m", "v"], ["Y"]))
    for name in ["s", "b", "m", "v"]:
        t = Tensor(name, (2,), DType.FLOAT32)
        t.data = 1.0  # Not ndarray
        t.is_initializer = True
        g2.tensors[name] = t
    fuse_batchnorm_into_gemm(g2)  # 300

    # 325: gemm_b not init or not ndarray
    g3 = Graph("G3")
    n_gemm3 = Node("Gemm", ["X", "W", "Bgemm"], ["gemm_out"])
    g3.nodes.append(n_gemm3)
    g3.nodes.append(Node("BatchNormalization", ["gemm_out", "s", "b", "m", "v"], ["Y"]))
    for name in ["s", "b", "m", "v"]:
        t = Tensor(name, (2,), DType.FLOAT32)
        t.data = np.ones(2, dtype=np.float32)
        t.is_initializer = True
        g3.tensors[name] = t
    w = Tensor("W", (2, 2), DType.FLOAT32)
    w.data = np.eye(2, dtype=np.float32)
    w.is_initializer = True
    g3.tensors["W"] = w
    bgemm = Tensor("Bgemm", (2,), DType.FLOAT32)
    bgemm.data = 1.0  # not ndarray
    bgemm.is_initializer = True
    g3.tensors["Bgemm"] = bgemm
    fuse_batchnorm_into_gemm(g3)  # 325

    # 374: out.name = bn_x in gemm fusion
    g4 = Graph("G4")
    n_gemm4 = Node("Gemm", ["X", "W"], ["gemm_out"])
    g4.nodes.append(n_gemm4)
    g4.nodes.append(Node("BatchNormalization", ["gemm_out", "s", "b", "m", "v"], ["Y"]))
    for name in ["s", "b", "m", "v"]:
        t = Tensor(name, (2,), DType.FLOAT32)
        t.data = np.ones(2, dtype=np.float32)
        t.is_initializer = True
        g4.tensors[name] = t
    g4.tensors["W"] = w

    class CustomOut:
        def __init__(self, name):
            self.name = name

    o = CustomOut("Y")
    g4.outputs = [o]
    fuse_batchnorm_into_gemm(g4)  # 374
    assert o.name == "gemm_out"


def test_fusion_conv_missing_lines_exact():
    # 415: num_consumers > 1
    g1 = Graph("G1c")
    n_conv = Node("Conv", ["X", "W"], ["conv_out"])
    g1.nodes.append(n_conv)
    g1.nodes.append(Node("BatchNormalization", ["conv_out", "s", "b", "m", "v"], ["Y"]))
    g1.nodes.append(Node("Abs", ["conv_out"], ["Z"]))  # second consumer
    fuse_batchnorm_into_conv(g1)  # 415

    # 436: not all ndarray
    g2 = Graph("G2c")
    n_conv2 = Node("Conv", ["X", "W"], ["conv_out"])
    g2.nodes.append(n_conv2)
    g2.nodes.append(Node("BatchNormalization", ["conv_out", "s", "b", "m", "v"], ["Y"]))
    for name in ["s", "b", "m", "v"]:
        t = Tensor(name, (2,), DType.FLOAT32)
        t.data = 1.0  # Not ndarray
        t.is_initializer = True
        g2.tensors[name] = t
    fuse_batchnorm_into_conv(g2)  # 436

    # 461: conv_b not init or not ndarray
    g3 = Graph("G3c")
    n_conv3 = Node("Conv", ["X", "W", "Bconv"], ["conv_out"])
    g3.nodes.append(n_conv3)
    g3.nodes.append(Node("BatchNormalization", ["conv_out", "s", "b", "m", "v"], ["Y"]))
    for name in ["s", "b", "m", "v"]:
        t = Tensor(name, (2,), DType.FLOAT32)
        t.data = np.ones(2, dtype=np.float32)
        t.is_initializer = True
        g3.tensors[name] = t
    w = Tensor("W", (2, 2, 1, 1), DType.FLOAT32)
    w.data = np.ones((2, 2, 1, 1), dtype=np.float32)
    w.is_initializer = True
    g3.tensors["W"] = w
    bconv = Tensor("Bconv", (2,), DType.FLOAT32)
    bconv.data = 1.0  # not ndarray
    bconv.is_initializer = True
    g3.tensors["Bconv"] = bconv
    fuse_batchnorm_into_conv(g3)  # 461

    # 506: out.name = bn_x in conv fusion
    g4 = Graph("G4c")
    n_conv4 = Node("Conv", ["X", "W"], ["conv_out"])
    g4.nodes.append(n_conv4)
    g4.nodes.append(Node("BatchNormalization", ["conv_out", "s", "b", "m", "v"], ["Y"]))
    for name in ["s", "b", "m", "v"]:
        t = Tensor(name, (2,), DType.FLOAT32)
        t.data = np.ones(2, dtype=np.float32)
        t.is_initializer = True
        g4.tensors[name] = t
    g4.tensors["W"] = w

    class CustomOut:
        def __init__(self, name):
            self.name = name

    o = CustomOut("Y")
    g4.outputs = [o]
    fuse_batchnorm_into_conv(g4)  # 506
    assert o.name == "conv_out"


def test_run_all_fusions():
    from onnx9000.optimizer.simplifier.passes.fusion import run_all_fusions

    g = Graph("G_all")
    g.nodes.append(Node("Gemm", ["X", "W"], ["Y"]))
    run_all_fusions(g)  # just hit the call


def test_fusion_gemm_missing_lines_exact_num_consumers():
    from onnx9000.optimizer.simplifier.passes.fusion import fuse_batchnorm_into_gemm

    g1 = Graph("G1_exact")
    n_gemm = Node("Gemm", ["X", "W"], ["gemm_out"])
    g1.nodes.append(n_gemm)
    n_bn = Node("BatchNormalization", ["gemm_out", "s", "b", "m", "v"], ["Y"])
    g1.nodes.append(n_bn)
    n_abs = Node("Abs", ["gemm_out"], ["Z"])
    g1.nodes.append(n_abs)

    fuse_batchnorm_into_gemm(g1)

    # 374: out.name = bn_x in gemm fusion
    g4 = Graph("G4_exact")
    n_gemm4 = Node("Gemm", ["X", "W"], ["gemm_out"])
    g4.nodes.append(n_gemm4)
    g4.nodes.append(Node("BatchNormalization", ["gemm_out", "s", "b", "m", "v"], ["Y"]))

    import numpy as np
    from onnx9000.core.dtypes import DType

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

    class CustomOut:
        def __init__(self, name):
            self.name = name

    o = CustomOut("Y")
    g4.outputs = [o]

    fuse_batchnorm_into_gemm(g4)  # 374


def test_fusion_conv_missing_lines_exact_num_consumers():
    from onnx9000.optimizer.simplifier.passes.fusion import fuse_batchnorm_into_conv

    g1 = Graph("G1_exact_conv")
    n_conv = Node("Conv", ["X", "W"], ["conv_out"])
    g1.nodes.append(n_conv)
    n_bn = Node("BatchNormalization", ["conv_out", "s", "b", "m", "v"], ["Y"])
    g1.nodes.append(n_bn)
    n_abs = Node("Abs", ["conv_out"], ["Z"])
    g1.nodes.append(n_abs)

    fuse_batchnorm_into_conv(g1)  # 415

    # 506: out.name = bn_x in conv fusion
    g4 = Graph("G4_exact_conv")
    n_conv4 = Node("Conv", ["X", "W"], ["conv_out"])
    g4.nodes.append(n_conv4)
    g4.nodes.append(Node("BatchNormalization", ["conv_out", "s", "b", "m", "v"], ["Y"]))

    import numpy as np
    from onnx9000.core.dtypes import DType

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

    class CustomOut:
        def __init__(self, name):
            self.name = name

    o = CustomOut("Y")
    g4.outputs = [o]

    fuse_batchnorm_into_conv(g4)  # 506


def test_map_aten_arange_no_op():
    from onnx9000.optimizer.simplifier.passes.fusion import map_aten_arange_to_range

    g = Graph("ATenMissing")
    n = Node("ATen", ["start", "end", "step", "dtype"], ["Y"], domain="org.pytorch.aten")
    # No operator attribute!
    g.nodes.append(n)
    map_aten_arange_to_range(g)  # 561?

    # Also test op_attr.value is not bytes
    n.attributes["operator"] = Attribute("operator", "STRING", "not_bytes")
    map_aten_arange_to_range(g)
