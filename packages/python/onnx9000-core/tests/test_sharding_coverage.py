"""Module docstring."""


def test_sharding_miss():
    """Docstring for D103."""
    from onnx9000.core.ir import Graph, Node, Tensor
    from onnx9000.core.sharding import (
        AutoShardingPass,
        PartitionSpec,
        SPMDLoweringPass,
        all_gather,
        all_reduce,
        all_to_all,
        reduce_scatter,
    )

    g = Graph("test")
    x = Tensor(name="x", shape=[4, 4], dtype=1)
    x.sharding = PartitionSpec("dp", "tp")
    w = Tensor(name="w", shape=[4, 4], dtype=1)
    w.sharding = PartitionSpec(None, "tp")
    w2 = Tensor(name="w2", shape=[4, 4], dtype=1)
    w2.sharding = PartitionSpec("tp", None)
    w3 = Tensor(name="w3", shape=[4, 4], dtype=1)
    w3.sharding = PartitionSpec("fsdp")
    w3.is_initializer = True

    y = Tensor(name="y", shape=[4, 4], dtype=1)

    n = Node("MatMul", inputs=[x, w], outputs=[y])
    n2 = Node("MatMul", inputs=[x, w2], outputs=[y])
    n3 = Node("MatMul", inputs=[x, w3], outputs=[y])
    n4 = Node("MatMul", inputs=[], outputs=[])  # len 0 test

    g.nodes.extend([n, n2, n3, n4])

    auto = AutoShardingPass()
    spmd = SPMDLoweringPass()

    auto.apply(g)
    spmd.apply(g)
