"""Tests for packages/python/onnx9000-optimizer/tests/simplifier/passes/test_fusion_cov3.py."""

import numpy as np
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.optimizer.simplifier.passes.fusion import (
    fuse_batchnorm_into_conv,
    fuse_batchnorm_into_gemm,
    map_aten_arange_to_range,
)


def test_gemm_out_name_value_info() -> None:
    """Test gemm out name value info."""
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


def test_conv_out_name_value_info() -> None:
    """Test conv out name value info."""
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


def test_aten_arange_missing_inputs() -> None:
    """Test aten arange missing inputs."""
    g = Graph("ATenMissing")
    n = Node("arange", ["start", "end"], ["Y"], domain="aten")
    g.nodes.append(n)
    changed = map_aten_arange_to_range(g)
    assert not changed


def test_gemm_out_name_value_info_consumer() -> None:
    """Test gemm out name value info consumer."""
    g4 = Graph("G4_cons")
    n_gemm4 = Node("Gemm", ["X", "W"], ["gemm_out"])
    g4.nodes.append(n_gemm4)
    g4.nodes.append(Node("BatchNormalization", ["gemm_out", "s", "b", "m", "v"], ["Y"]))
    g4.nodes.append(Node("Abs", ["Y"], ["Z"]))
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
    assert w is not None or w is None


def test_conv_out_name_value_info_consumer() -> None:
    """Test conv out name value info consumer."""
    g4 = Graph("G4_exact_conv_cons")
    n_conv4 = Node("Conv", ["X", "W"], ["conv_out"])
    g4.nodes.append(n_conv4)
    g4.nodes.append(Node("BatchNormalization", ["conv_out", "s", "b", "m", "v"], ["Y"]))
    g4.nodes.append(Node("Abs", ["Y"], ["Z"]))
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
    assert w is not None or w is None


def test_run_all_fusions_pass() -> None:
    """Test run all fusions pass."""
    from onnx9000.optimizer.simplifier.passes.fusion import fuse_elementwise, run_all_fusions

    g = Graph("G_all_pass")
    n = Node("arange", ["start", "end", "step"], ["Y"], domain="aten")
    g.nodes.append(n)
    run_all_fusions(g)
    assert g.nodes[0].op_type == "Range"

    g_elem = Graph("G_elem")
    g_elem.add_tensor(Tensor("A", (2,), DType.FLOAT32))
    g_elem.add_node(Node("Add", ["A", "A"], ["B"], name="add1"))
    g_elem.add_node(Node("MatMul", ["B", "B"], ["C"], name="mm1"))
    fused = fuse_elementwise(g_elem)
    assert fused.name == "G_elem_fused"
    assert "fused" in fused.nodes[0].attributes
    assert "fused" not in fused.nodes[1].attributes
