"""Tests the ir more module functionality."""

from onnx9000.core.ir import *


def test_ir_more() -> None:
    """Tests the ir more functionality."""
    t = Tensor("t_base")
    assert repr(t) == "ir.Tensor(name=t_base)"
    h = hash(t)
    assert h == object.__hash__(t)
    c = Constant("c", values=b"abc", shape=(3,))
    c_copy = c.copy()
    assert isinstance(c_copy, Constant)
    assert c_copy.data == b"abc"
    v = Variable("v")
    n1 = Node("n1", outputs=[v])
    n2 = Node("n2", inputs=[v])
    assert v in n1.outputs
    assert v in n2.inputs
    v.clear_inputs()
    assert v not in n1.outputs
    v.clear_outputs()
    assert v not in n2.inputs
    assert hash(n1) == object.__hash__(n1)
    g = Graph("g")
    v_dup1 = Variable("v_dup")
    v_dup2 = Variable("v_dup")
    g.add_tensor(v_dup1)
    g.add_tensor(v_dup2)
    assert v_dup2.name == "v_dup_1"
    n_dup1 = Node("Op", name="n_dup")
    n_dup2 = Node("Op", name="n_dup")
    g.add_node(n_dup1)
    g.add_node(n_dup2)
    assert n_dup2.name == "n_dup_1"
    n_noname = Node("MyOp")
    n_noname2 = Node("MyOp")
    g.add_node(n_noname)
    g.add_node(n_noname2)
    assert n_noname.name == "MyOp"
    assert n_noname2.name == "MyOp_1"
    g2 = Graph("g")
    assert g != g2
    assert n_dup1 == n_dup2
    assert n_dup1 != "string"
    assert g.get_node("missing") is None
    g.print_visualizer()


def test_ir_edge_cases() -> None:
    """Tests the ir edge cases functionality."""
    n1 = Node("Op", inputs=["in1"])
    n2 = Node("Op", inputs=["in1", "in2"])
    assert n1 != n2
    n3 = Node("Op", inputs=["in1"], outputs=["out1"])
    n4 = Node("Op", inputs=["in1"], outputs=["out1", "out2"])
    assert n3 != n4
    n5 = Node("Op", inputs=["in1"], attributes={"a": Attribute("a", value=1)})
    n6 = Node("Op", inputs=["in1"], attributes={"a": Attribute("a", value=2)})
    assert n5 != n6
    g = Graph("g")
    n_empty1 = Node("Op", name="")
    n_empty2 = Node("Op", name="")
    g.add_node(n_empty1)
    g.add_node(n_empty2)
    v_empty1 = Variable("")
    v_empty2 = Variable("")
    g.add_tensor(v_empty1)
    g.add_tensor(v_empty2)
    g_other_name = Graph("g2")
    assert g != g_other_name
    g_other_len = Graph("g")
    g_other_len.add_node(Node("Op", name="op1"))
    assert g != g_other_len
    g_other_nodes = Graph("g")
    g_other_nodes.add_node(Node("Op1", name="op1"))
    g_other_nodes2 = Graph("g")
    g_other_nodes2.add_node(Node("Op2", name="op1"))
    assert g_other_nodes != g_other_nodes2


def test_sparse_ir():
    """Docstring for D103."""
    from onnx9000.core.ir import Attribute, Graph, SparseTensor, Tensor

    st = SparseTensor("sp")
    assert Attribute.infer_type(st) == "SPARSE_TENSOR"
    assert Attribute.infer_type([st]) == "SPARSE_TENSORS"
    assert Attribute.infer_type(Graph("g")) == "GRAPH"
    assert Attribute.infer_type([Graph("g")]) == "GRAPHS"
    assert repr(st) == "ir.SparseTensor(name=sp, shape=(), format=COO)"
    st2 = st.copy()
    assert st2.name == "sp"
    st3 = Tensor.copy(st)
    assert st3.name == "sp"
