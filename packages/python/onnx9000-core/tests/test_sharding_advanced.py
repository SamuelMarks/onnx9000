"""Tests for sharding advanced."""

from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.sharding import PartitionSpec, SPMDLoweringPass


def test_tp_sliced() -> None:
    """Verify that tp sliced behaves as expected."""
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


def test_tp_allreduce() -> None:
    """Verify that tp allreduce behaves as expected."""
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


def test_ep() -> None:
    """Verify that ep behaves as expected."""
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


def test_pp() -> None:
    """Verify that pp behaves as expected."""
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


def test_cp() -> None:
    """Verify that cp behaves as expected."""
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


def test_fsdp() -> None:
    """Verify that fsdp behaves as expected."""
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
