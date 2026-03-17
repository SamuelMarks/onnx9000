import numpy as np
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.optimizer.simplifier.api import simplify


def test_constant_folding_extras() -> None:
    g = Graph("test")
    t = Tensor(
        "in0",
        shape=(4,),
        dtype=DType.FLOAT32,
        data=np.array([1, 2, 3, 4], dtype=np.float32),
        is_initializer=True,
    )
    g.add_tensor(t)
    g.initializers.append("in0")
    g.add_node(Node("Split", ["in0"], ["out1", "out2"], {}, "n1"))
    g.outputs = ["out1", "out2"]
    t2 = Tensor(
        "in1",
        shape=(2,),
        dtype=DType.INT64,
        data=np.array([2, 2], dtype=np.int64),
        is_initializer=True,
    )
    g.add_tensor(t2)
    g.initializers.append("in1")
    g.add_node(Node("ConstantOfShape", ["in1"], ["out3"], {"value": [1.0]}, "n2"))
    val_t = Tensor("val", shape=(1,), dtype=DType.FLOAT32, data=np.array([2.0], dtype=np.float32))
    g.add_node(Node("ConstantOfShape", ["in1"], ["out4"], {"value": val_t}, "n3"))
    t_conv = Tensor(
        "in_conv",
        shape=(1, 1, 3, 3),
        dtype=DType.FLOAT32,
        data=np.zeros((1, 1, 3, 3), dtype=np.float32),
        is_initializer=True,
    )
    g.add_tensor(t_conv)
    g.initializers.append("in_conv")
    g.add_node(Node("Conv", ["in_conv", "in_conv"], ["out5"], {}, "n4"))
    g.outputs.extend(["out3", "out4", "out5"])
    simplify(g)


def test_dce_transpose_fused() -> None:
    g = Graph("test")
    g.inputs = ["in0"]
    g.add_node(Node("Transpose", ["in0"], ["mid"], {"perm": [1, 0, 2]}, "n1"))
    g.add_node(Node("Transpose", ["mid"], ["out"], {"perm": [2, 1, 0]}, "n2"))
    g.outputs = ["out"]
    g_sim = simplify(g)
    assert len(g_sim.nodes) == 1
    assert g_sim.nodes[0].op_type == "Transpose"
    assert g_sim.nodes[0].attributes["perm"] == [2, 0, 1]
