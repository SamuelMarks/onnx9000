import pytest
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.dtypes import DType
from onnx9000.core.parser.memory import plan_memory


def test_plan_memory_full():
    g = Graph("test")
    t_in = Tensor("in1", (1,), DType.FLOAT32, is_initializer=False)
    t_w = Tensor("w", (1,), DType.FLOAT32, is_initializer=True)
    t_out1 = Tensor("out1", (1,), DType.FLOAT32)
    t_out2 = Tensor("out2", (1,), DType.FLOAT32)
    t_final = Tensor("final", (1,), DType.FLOAT32)
    g.add_tensor(t_in)
    g.add_tensor(t_w)
    g.add_tensor(t_out1)
    g.add_tensor(t_out2)
    g.add_tensor(t_final)
    g.inputs.append("in1")
    g.initializers.append("w")
    g.outputs.append("final")
    n1 = Node("Add", ["in1", "w"], ["out1"], {})
    n2 = Node("Relu", ["out1"], ["out2"], {})
    n3 = Node("Abs", ["out2"], ["final"], {})
    g.nodes.extend([n1, n2, n3])
    plan_memory(g)
    assert t_out1.buffer_id is not None
    assert t_out2.buffer_id is not None
    assert t_final.buffer_id is not None


def test_plan_memory_overlap():
    g = Graph("overlap")
    t_in = Tensor("in1", (1,), DType.FLOAT32)
    t_o1 = Tensor("o1", (1,), DType.FLOAT32)
    t_o2 = Tensor("o2", (1,), DType.FLOAT32)
    g.add_tensor(t_in)
    g.add_tensor(t_o1)
    g.add_tensor(t_o2)
    g.inputs.append("in1")
    g.outputs.extend(["o1", "o2"])
    n1 = Node("Split", ["in1"], ["o1", "o2"], {})
    g.nodes.append(n1)
    plan_memory(g)
    assert t_o1.buffer_id != t_o2.buffer_id
