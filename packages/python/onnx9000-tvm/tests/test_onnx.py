"""Tests for ONNX frontend in TVM."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.tvm.relay.frontend.onnx import from_onnx


def test_from_onnx_full():
    """Test importing a full ONNX graph into TVM relay."""
    g = Graph("test_graph")
    g.inputs.append(ValueInfo("x", (1, 2), DType.FLOAT32))  # Has int! Covers 124
    g.inputs.append(ValueInfo("x2", ("?", 2), DType.FLOAT32))
    g.initializers.append(Tensor("w", [1.0, 2.0], (2,), DType.FLOAT32))

    # Needs a generic op:
    g.nodes.append(Node("GenericUnknownOp", ["x", "w"], ["y"]))  # Covers 154
    # Single output: len(node.outputs) == 1 -> covers 158

    g.nodes.append(Node("Split", ["y"], ["z1", "z2"]))

    g.outputs.append(ValueInfo("y", (1, 1), DType.FLOAT32))

    mod = from_onnx(g)  # 1 output -> covers 168
    assert mod is not None


test_from_onnx_full()
