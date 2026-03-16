import numpy as np
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.optimizer.simplifier.api import simplify


def test_simplify_purely_arithmetic_graph():
    g = Graph("mock")
    val_a = np.array([2], dtype=np.float32)
    val_b = np.array([3], dtype=np.float32)
    val_c = np.array([4], dtype=np.float32)
    t_a = Tensor("a", shape=(1,), dtype=DType.FLOAT32, data=val_a, is_initializer=True)
    t_b = Tensor("b", shape=(1,), dtype=DType.FLOAT32, data=val_b, is_initializer=True)
    t_c = Tensor("c", shape=(1,), dtype=DType.FLOAT32, data=val_c, is_initializer=True)
    g.add_tensor(t_a)
    g.add_tensor(t_b)
    g.add_tensor(t_c)
    g.initializers = ["a", "b", "c"]
    g.add_node(Node("Mul", ["b", "c"], ["d"], {}, "mul"))
    g.add_node(Node("Add", ["a", "d"], ["out"], {}, "add"))
    g.outputs = ["out"]
    g_sim = simplify(g)
    assert len(g_sim.nodes) == 1
    assert g_sim.nodes[0].op_type == "Constant"
    np.testing.assert_array_equal(
        g_sim.nodes[0].attributes["value"], np.array([14], dtype=np.float32)
    )
