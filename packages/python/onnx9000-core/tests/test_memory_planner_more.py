import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Constant, Graph, Node, Tensor, ValueInfo
from onnx9000.core.memory_planner import simulate_memory_plan


def test_memory_plan_more():
    g = Graph("g1")
    g.inputs.append(ValueInfo("in", [100], DType.FLOAT32))
    g.tensors["in"] = Tensor("in", shape=[100], dtype=DType.FLOAT32)

    g.tensors["out"] = Tensor("out", shape=[100], dtype=DType.FLOAT32)
    g.tensors["c"] = Constant("c", values=b"0" * 400, shape=[100], dtype=DType.FLOAT32)

    g.nodes.append(Node("Add", ["in", "c"], ["out"]))
    g.nodes.append(Node("Relu", ["out"], ["out2"]))

    g.tensors["out2"] = Tensor("out2", shape=[100], dtype=DType.FLOAT32)

    arena = simulate_memory_plan(g, strategy="best_fit")
    assert arena.peak_memory > 0


def test_planner_input_used():
    from onnx9000.core.ir import Graph, Node, Tensor
    from onnx9000.core.memory_planner import simulate_memory_plan

    g = Graph("g")
    g.inputs.append(Tensor("in", [1], "float32"))
    g.add_node(Node("Add", ["in", "in"], ["out"]))
    g.outputs.append(Tensor("out", [1], "float32"))
    g.tensors["in"] = g.inputs[0]
    g.tensors["out"] = g.outputs[0]
    simulate_memory_plan(g)
