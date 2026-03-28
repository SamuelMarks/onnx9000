"""Module providing functionality for test_dce_cov2."""

import numpy as np
import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor, ValueInfo
from onnx9000.optimizer.simplifier.passes.dce import DCEPass, IdentityEliminationPass


def test_dce_preserve_nodes():
    """Test dce preserve nodes."""
    g = Graph("TestPreserve")
    g.inputs = ["X"]
    g.outputs = ["Y"]
    g.tensors["X"] = Tensor("X", (2,), DType.FLOAT32)
    g.tensors["Y"] = Tensor("Y", (2,), DType.FLOAT32)
    g.tensors["Z"] = Tensor("Z", (2,), DType.FLOAT32)

    n1 = Node("Relu", inputs=["X"], outputs=["Y"], name="n_used")
    n2 = Node("Relu", inputs=["X"], outputs=["Z"], name="n_preserve")

    g.nodes.extend([n1, n2])

    # Run with preserve
    dce = DCEPass(nodes_to_preserve={"n_preserve"})
    dce.run(g)

    ops = [n.name for n in g.nodes]
    assert "n_used" in ops
    assert "n_preserve" in ops


def test_identity_subgraphs():
    """Test identity subgraphs."""
    g = Graph("TestIdSub")
    sub = Graph("Sub")
    sub.tensors["A"] = Tensor("A", (1,), DType.FLOAT32)
    sub.tensors["B"] = Tensor("B", (1,), DType.FLOAT32)
    sub.nodes.append(Node("Identity", inputs=["A"], outputs=["B"]))

    n_if = Node("If", inputs=["cond"], outputs=["Y"])
    n_if.attributes["then_branch"] = Attribute("then_branch", None, sub)
    g.nodes.append(n_if)

    ie = IdentityEliminationPass()
    changed = ie.run(g)
    assert changed
    assert len(sub.nodes) == 0


def test_identity_folding():
    """Test identity folding."""
    g = Graph("TestFold")

    t_c1 = Tensor("C1", (1,), DType.FLOAT32)
    t_c1.data = np.array([2.0], dtype=np.float32)
    t_c1.is_initializer = True
    g.tensors["C1"] = t_c1

    t_c2 = Tensor("C2", (1,), DType.FLOAT32)
    t_c2.data = np.array([3.0], dtype=np.float32)
    t_c2.is_initializer = True
    g.tensors["C2"] = t_c2

    g.tensors["X"] = Tensor("X", (1,), DType.FLOAT32)
    g.tensors["M"] = Tensor("M", (1,), DType.FLOAT32)
    g.tensors["Y"] = Tensor("Y", (1,), DType.FLOAT32)

    n1 = Node("Add", inputs=["X", "C1"], outputs=["M"])
    n2 = Node("Add", inputs=["M", "C2"], outputs=["Y"], name="add2")
    g.nodes.extend([n1, n2])

    ie = IdentityEliminationPass()
    changed = ie.run(g)
    assert changed

    # Mul folding
    g2 = Graph("TestFoldMul")

    t2_c1 = Tensor("C1", (1,), DType.FLOAT32)
    t2_c1.data = np.array([2.0], dtype=np.float32)
    t2_c1.is_initializer = True
    g2.tensors["C1"] = t2_c1

    t2_c2 = Tensor("C2", (1,), DType.FLOAT32)
    t2_c2.data = np.array([3.0], dtype=np.float32)
    t2_c2.is_initializer = True
    g2.tensors["C2"] = t2_c2

    g2.tensors["X"] = Tensor("X", (1,), DType.FLOAT32)
    g2.tensors["M"] = Tensor("M", (1,), DType.FLOAT32)
    g2.tensors["Y"] = Tensor("Y", (1,), DType.FLOAT32)

    n3 = Node("Mul", inputs=["X", "C1"], outputs=["M"])
    n4 = Node("Mul", inputs=["M", "C2"], outputs=["Y"], name="mul2")
    g2.nodes.extend([n3, n4])

    ie2 = IdentityEliminationPass()
    changed2 = ie2.run(g2)
    assert changed2


def test_mul_add_distribute():
    """Test mul add distribute."""
    g = Graph("TestDist")
    t_c1 = Tensor("C1", (1,), DType.FLOAT32)
    t_c1.data = np.array([2.0])
    t_c1.is_initializer = True
    g.tensors["C1"] = t_c1

    t_c2 = Tensor("C2", (1,), DType.FLOAT32)
    t_c2.data = np.array([3.0])
    t_c2.is_initializer = True
    g.tensors["C2"] = t_c2

    g.tensors["X"] = Tensor("X", (1,), DType.FLOAT32)
    g.tensors["M"] = Tensor("M", (1,), DType.FLOAT32)
    g.tensors["Y"] = Tensor("Y", (1,), DType.FLOAT32)

    n1 = Node("Add", inputs=["X", "C1"], outputs=["M"])
    n2 = Node("Mul", inputs=["M", "C2"], outputs=["Y"], name="mul_node")
    g.nodes.extend([n1, n2])

    ie = IdentityEliminationPass()
    changed = ie.run(g)
    assert changed
    ops = [n.op_type for n in g.nodes]
    assert ops.count("Mul") == 1
    assert ops.count("Add") == 2
