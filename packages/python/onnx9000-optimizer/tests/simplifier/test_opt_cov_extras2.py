import logging

import numpy as np
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.optimizer.simplifier.api import simplify

logging.basicConfig(level=logging.DEBUG)


def test_split() -> None:
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
    g_sim = simplify(g)
    print(g_sim.nodes)
