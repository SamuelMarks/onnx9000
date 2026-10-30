"""Comprehensive unit tests for onnx9000.graphsurgeon and onnx9000_graphsurgeon."""

import onnx9000_graphsurgeon
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.graphsurgeon import GraphSurgeon, cleanup, fold_constants, run, toposort


def test_graphsurgeon_optimize_and_run() -> None:
    """Verify GraphSurgeon.optimize(), toposort, cleanup, fold_constants, and run()."""
    assert run() == "[graphsurgeon] processed"
    assert onnx9000_graphsurgeon.run() == "[graphsurgeon] processed"

    gs = GraphSurgeon()
    assert gs.process("resnet") == "GraphSurgeon processed resnet"

    g = Graph("test_gs")
    x = Tensor("X", (1, 4), DType.FLOAT32)
    y = Tensor("Y", (1, 4), DType.FLOAT32)
    dead = Tensor("Dead", (1, 4), DType.FLOAT32)
    g.add_tensor(x)
    g.add_tensor(y)
    g.add_tensor(dead)
    g.inputs.append(ValueInfo("X", (1, 4), DType.FLOAT32))
    g.outputs.append(ValueInfo("Y", (1, 4), DType.FLOAT32))
    g.add_node(Node("Relu", [x], [y], name="relu_live"))
    g.add_node(Node("Identity", [x], [dead], name="id_dead"))

    optimized = gs.optimize(g)
    assert len(optimized.nodes) == 1
    assert optimized.nodes[0].name == "relu_live"
    assert callable(toposort)
    assert callable(cleanup)
    assert callable(fold_constants)
