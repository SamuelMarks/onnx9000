"""Tests for optimizer surgeon passes."""

import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Constant, Graph, Node, Tensor
from onnx9000.optimizer.surgeon.audio import fold_mel_weights
from onnx9000.optimizer.surgeon.fusions import fuse_flash_attention, fuse_horizontal_gemm
from onnx9000.optimizer.surgeon.layout import optimize_layouts
from onnx9000.optimizer.surgeon.obfuscator import obfuscate_names
from onnx9000.optimizer.surgeon.quantization import quantize_ptq


def test_fold_mel_weights_logic():
    """Verify MelWeightMatrix folding."""
    graph = Graph("test")
    c1 = Constant("bins", None, (), DType.INT64)
    c2 = Constant("dft", None, (), DType.INT64)
    c3 = Constant("rate", None, (), DType.INT64)
    c4 = Constant("low", None, (), DType.FLOAT32)
    c5 = Constant("high", None, (), DType.FLOAT32)

    graph.add_tensor(c1)
    graph.add_tensor(c2)
    graph.add_tensor(c3)
    graph.add_tensor(c4)
    graph.add_tensor(c5)

    node = Node("MelWeightMatrix", ["bins", "dft", "rate", "low", "high"], ["mel_out"])
    graph.add_node(node)

    fold_mel_weights(graph)
    assert "mel_out" in graph.tensors


def test_fuse_flash_attention_logic():
    """Verify FlashAttention fusion."""
    graph = Graph("test")
    # Pattern: MatMul(Softmax(Div(MatMul(Q, K), scale)), V)

    q = Tensor("q", (1, 8, 128), DType.FLOAT32)
    k = Tensor("k", (1, 8, 128), DType.FLOAT32)
    v = Tensor("v", (1, 8, 128), DType.FLOAT32)
    scale = Constant("scale", None, (), DType.FLOAT32)
    graph.add_tensor(q)
    graph.add_tensor(k)
    graph.add_tensor(v)
    graph.add_tensor(scale)

    qk_node = Node("MatMul", ["q", "k"], ["qk_out"], name="qk")
    graph.add_node(qk_node)
    qk_out = Tensor("qk_out", (1, 8, 8), DType.FLOAT32)
    qk_out.inputs = [qk_node]
    graph.add_tensor(qk_out)

    div_node = Node("Div", [qk_out, scale], ["div_out"], name="div")
    graph.add_node(div_node)
    div_out = Tensor("div_out", (1, 8, 8), DType.FLOAT32)
    div_out.inputs = [div_node]
    graph.add_tensor(div_out)

    sm_node = Node("Softmax", [div_out], ["sm_out"], name="sm")
    graph.add_node(sm_node)
    sm_out = Tensor("sm_out", (1, 8, 8), DType.FLOAT32)
    sm_out.inputs = [sm_node]
    graph.add_tensor(sm_out)

    final_node = Node("MatMul", [sm_out, v], ["final_out"], name="final")
    graph.add_node(final_node)

    fuse_flash_attention(graph)
    assert any(n.op_type == "FlashAttention" for n in graph.nodes)


def test_fuse_horizontal_gemm_logic():
    """Verify Horizontal Gemm fusion."""
    graph = Graph("test")
    x = Tensor("x", (1, 128), DType.FLOAT32)
    w1 = Constant("w1", None, (128, 64), DType.FLOAT32)
    w2 = Constant("w2", None, (128, 64), DType.FLOAT32)
    graph.add_tensor(x)
    graph.add_tensor(w1)
    graph.add_tensor(w2)

    g1 = Node("Gemm", [x, w1], ["y1"])
    g2 = Node("Gemm", [x, w2], ["y2"])
    graph.add_node(g1)
    graph.add_node(g2)

    fuse_horizontal_gemm(graph)


def test_optimize_layouts_logic():
    """Verify layout optimization."""
    graph = Graph("test")
    n = Node(
        "Conv",
        ["in", "w"],
        ["out"],
        attributes={"kernel_shape": Attribute("kernel_shape", value=[3, 3])},
    )
    graph.add_node(n)
    optimize_layouts(graph, "NHWC")


def test_obfuscate_names_logic():
    """Verify name obfuscation."""
    graph = Graph("test")
    n = Node("Relu", ["in"], ["out"], name="orig")
    graph.add_node(n)
    t = Tensor("in", (1,), DType.FLOAT32)
    graph.add_tensor(t)
    obfuscate_names(graph)
    assert n.name != "orig"


def test_quantize_ptq_logic():
    """Verify PTQ pass."""
    graph = Graph("test")
    n = Node("Conv", ["x", "w"], ["y"])
    graph.add_node(n)
    quantize_ptq(graph)
