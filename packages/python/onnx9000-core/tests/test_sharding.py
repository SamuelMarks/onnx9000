"""Module docstring."""

from onnx9000.core.ir import Graph, Tensor
from onnx9000.core.sharding import (
    AutoShardingPass,
    SPMDLoweringPass,
    all_gather,
    all_reduce,
    all_to_all,
    reduce_scatter,
)


def test_sharding():
    """Docstring for D103."""
    x = Tensor(name="x", shape=[1], dtype=1)
    x.sharding = (None, "tp_x")

    assert all_reduce(x).name == "AllReduce_out"
    assert all_gather(x).name == "AllGather_out"
    assert reduce_scatter(x).name == "ReduceScatter_out"
    assert all_to_all(x).name == "AllToAll_out"

    auto = AutoShardingPass()
    spmd = SPMDLoweringPass()
    g = Graph("test")
    assert auto.apply(g) is g
    assert spmd.apply(g) is g
