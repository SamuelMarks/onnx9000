import pytest
from onnx9000.core.ir import Constant, Graph, Node, Variable
from onnx9000.optimizer.surgeon.fusions import fuse_horizontal_gemm


def test_fuse_gemm_concats_no_data():
    g = Graph("test")
    in_t = Variable("in")
    g.add_tensor(in_t)

    # Needs 2 fusible Gemms.
    w1 = Constant("w1", shape=(2, 2), values=None)  # no data
    w2 = Constant("w2", shape=(2, 2), values=None)  # no data
    g.add_tensor(w1)
    g.add_tensor(w2)

    out1 = Variable("out1")
    out2 = Variable("out2")
    g.add_tensor(out1)
    g.add_tensor(out2)

    n1 = Node("Gemm", name="n1", inputs=["in", "w1"], outputs=["out1"])
    n2 = Node("Gemm", name="n2", inputs=["in", "w2"], outputs=["out2"])
    g.add_node(n1)
    g.add_node(n2)

    fuse_horizontal_gemm(g)


def test_fuse_gemm_concats_with_data():
    g = Graph("test")
    in_t = Variable("in")
    g.add_tensor(in_t)

    # Needs 2 fusible Gemms.
    w1 = Constant("w1", shape=(2, 2), values=b"abcd")
    w2 = Constant("w2", shape=(2, 2), values=b"efgh")
    g.add_tensor(w1)
    g.add_tensor(w2)

    out1 = Variable("out1")
    out2 = Variable("out2")
    g.add_tensor(out1)
    g.add_tensor(out2)

    n1 = Node("Gemm", name="n1", inputs=["in", "w1"], outputs=["out1"])
    n2 = Node("Gemm", name="n2", inputs=["in", "w2"], outputs=["out2"])
    g.add_node(n1)
    g.add_node(n2)

    fuse_horizontal_gemm(g)
