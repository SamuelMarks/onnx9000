"""Module docstring."""


def test_sharding_advanced_coverage():
    """Docstring for D103."""
    from onnx9000.core.ir import Graph, Node, Tensor
    from onnx9000.core.sharding import (
        PartitionSpec,
        SPMDLoweringPass,
        all_gather,
        all_reduce,
        all_to_all,
        reduce_scatter,
    )

    g = Graph("test")
    x = Tensor(name="x", shape=[4, 4], dtype=1)

    # 2. Expert Parallelism
    w_ep2 = Tensor(name="w_ep2", shape=[4, 4], dtype=1)
    w_ep2.sharding = PartitionSpec("ep", None, None)
    n_ep2 = Node("MatMul", inputs=[x, w_ep2], outputs=[Tensor("y2b", [4, 4], 1)])
    g.nodes.append(n_ep2)

    # 3. Context Parallelism
    n_cp = Node("FlashAttention", inputs=[x], outputs=[Tensor("y3", [4, 4], 1)])
    x_cp = Tensor(name="x_cp", shape=[4, 4], dtype=1)
    x_cp.sharding = PartitionSpec(None, "cp", None)
    n_cp.inputs = [x_cp]
    g.nodes.append(n_cp)

    # 4. Pipeline Parallelism
    n_pp = Node("Add", inputs=[x, x], outputs=[Tensor("y4", [4, 4], 1)])
    n_pp.outputs[0].sharding = PartitionSpec("pp")
    g.nodes.append(n_pp)

    spmd = SPMDLoweringPass()
    spmd.apply(g)

    ops = [n.op_type for n in g.nodes]
    assert "AllToAll" in ops
    assert "Recv" in ops
    assert "Send" in ops
