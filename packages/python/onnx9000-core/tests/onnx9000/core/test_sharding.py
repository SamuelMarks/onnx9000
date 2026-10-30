"""Unit tests for distributed sharding passes and collective ops in `onnx9000.core.sharding`."""

from onnx9000.core.dtypes import DType
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


def test_sharding_passes_and_collectives() -> None:
    """Verify PartitionSpec, AutoShardingPass, SPMDLoweringPass, and collective ops."""
    spec = PartitionSpec("tp", None)
    assert spec == ("tp", None)

    g = Graph("sharded")
    a = Tensor("a", (4, 4), DType.FLOAT32)
    b = Tensor("b", (4, 4), DType.FLOAT32)
    b.sharding = spec
    out = Tensor("out", (4, 4), DType.FLOAT32)
    g.add_node(Node("MatMul", inputs=[a, b], outputs=[out], name="mm"))

    AutoShardingPass().apply(g)
    assert out.sharding == (None, None)

    SPMDLoweringPass().apply(g)
    assert any(n.op_type == "AllReduce" for n in g.nodes)

    x = Tensor("x", (4, 4), DType.FLOAT32)
    assert isinstance(all_reduce(x), Tensor)
    assert isinstance(all_gather(x), Tensor)
    assert isinstance(reduce_scatter(x), Tensor)
    assert isinstance(all_to_all(x), Tensor)
