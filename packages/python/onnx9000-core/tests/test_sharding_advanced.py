"""Tests for sharding advanced."""

import pytest
from onnx9000.core.ir import Attribute, Graph, Node, Tensor
from onnx9000.core.sharding import PartitionSpec, SPMDLoweringPass


def test_tp_sliced():
    """Docstring for D103."""
    graph = Graph("test")
    x = Tensor("x", shape=(4, 4), dtype=1)
    w = Tensor("w", shape=(4, 4), dtype=1)
    w.sharding = PartitionSpec(None, "tp")
    out = Tensor("out", shape=(4, 4), dtype=1)
    node = Node("MatMul", inputs=[x, w], outputs=[out])
    graph.nodes.append(node)

    spmd = SPMDLoweringPass()
    spmd.apply(graph)
    assert graph.nodes[0].outputs[0].sharding == PartitionSpec(None, "tp")


def test_tp_allreduce():
    """Docstring for D103."""
    graph = Graph("test")
    x = Tensor("x", shape=(4, 4), dtype=1)
    w = Tensor("w", shape=(4, 4), dtype=1)
    w.sharding = PartitionSpec("tp", None)
    out = Tensor("out", shape=(4, 4), dtype=1)
    node = Node("MatMul", inputs=[x, w], outputs=[out])
    graph.nodes.append(node)

    spmd = SPMDLoweringPass()
    spmd.apply(graph)

    assert len(graph.nodes) == 2
    assert graph.nodes[0].op_type == "MatMul"
    assert graph.nodes[1].op_type == "AllReduce"
    assert graph.nodes[1].attributes["op"].value == "sum"


def test_ep():
    """Docstring for D103."""
    graph = Graph("test")
    x = Tensor("x", shape=(4, 4), dtype=1)
    x.sharding = PartitionSpec("ep", None, None)
    out = Tensor("out", shape=(4, 4), dtype=1)
    node = Node("CustomEP", inputs=[x], outputs=[out])
    graph.nodes.append(node)

    spmd = SPMDLoweringPass()
    spmd.apply(graph)

    assert len(graph.nodes) == 3
    assert graph.nodes[0].op_type == "AllToAll"
    assert graph.nodes[1].op_type == "CustomEP"
    assert graph.nodes[2].op_type == "AllToAll"


def test_pp():
    """Docstring for D103."""
    graph = Graph("test")
    x = Tensor("x", shape=(4, 4), dtype=1)
    out = Tensor("out", shape=(4, 4), dtype=1)
    out.sharding = PartitionSpec("pp")
    node = Node("MatMul", inputs=[x, x], outputs=[out])
    graph.nodes.append(node)

    spmd = SPMDLoweringPass()
    spmd.apply(graph)

    assert len(graph.nodes) == 3
    assert graph.nodes[0].op_type == "MatMul"
    assert graph.nodes[1].op_type == "Send"
    assert graph.nodes[2].op_type == "Recv"


def test_cp():
    """Docstring for D103."""
    graph = Graph("test")
    q = Tensor("q", shape=(4, 4), dtype=1)
    q.sharding = PartitionSpec(None, "cp", None)
    k = Tensor("k", shape=(4, 4), dtype=1)
    v = Tensor("v", shape=(4, 4), dtype=1)
    out = Tensor("out", shape=(4, 4), dtype=1)
    node = Node("FlashAttention", inputs=[q, k, v], outputs=[out])
    graph.nodes.append(node)

    spmd = SPMDLoweringPass()
    spmd.apply(graph)

    assert len(graph.nodes) == 3
    assert graph.nodes[0].op_type == "Recv"
    assert graph.nodes[1].op_type == "FlashAttention"
    assert graph.nodes[2].op_type == "Send"


def test_fsdp():
    """Docstring for D103."""
    graph = Graph("test")
    x = Tensor("x", shape=(4, 4), dtype=1)
    w = Tensor("w", shape=(4, 4), dtype=1, is_initializer=True)
    w.sharding = PartitionSpec("fsdp")
    out = Tensor("out", shape=(4, 4), dtype=1)
    node = Node("MatMul", inputs=[x, w], outputs=[out])
    graph.nodes.append(node)

    spmd = SPMDLoweringPass()
    spmd.apply(graph)

    assert len(graph.nodes) == 3
    assert graph.nodes[0].op_type == "AllGather"
    assert graph.nodes[1].op_type == "MatMul"
    assert graph.nodes[2].op_type == "Discard"
