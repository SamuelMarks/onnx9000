import pytest
import numpy as np
from onnx9000.frontends.jit.hasher import hash_graph
from onnx9000.core.ir import Graph, Node, Tensor, DynamicDim
from onnx9000.core.dtypes import DType


def test_hash_graph():
    g = Graph("test")
    t1 = Tensor("in1", (2, 2), DType.FLOAT32)
    t2 = Tensor("out", (2, 2), DType.FLOAT32)
    t3 = Tensor("w", (2, 2), DType.FLOAT32, is_initializer=True)
    t3.data = np.array([1, 2, 3, 4], dtype=np.float32)
    g.add_tensor(t1)
    g.add_tensor(t2)
    g.add_tensor(t3)
    n = Node("Add", ["in1", "w"], ["out"], {})
    g.nodes.append(n)
    h = hash_graph(g)
    assert isinstance(h, str)
    assert len(h) == 64
    h2 = hash_graph(g)
    assert h == h2
