import pytest
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.dtypes import DType
from onnx9000.core.parser.memory import plan_memory
from onnx9000.core.exceptions import CompilationError


def test_memory_plan_duplicate_output():
    g = Graph("test")
    g.inputs.append("duplicate_out")
    n = Node(op_type="Relu", inputs=[], outputs=["duplicate_out"], attributes={})
    g.nodes.append(n)
    plan_memory(g)


def test_memory_plan_uncreated_input():
    g = Graph("test2")
    n = Node(op_type="Relu", inputs=["unknown_in"], outputs=["out"], attributes={})
    g.nodes.append(n)
    with pytest.raises(CompilationError, match="used before creation"):
        plan_memory(g)
