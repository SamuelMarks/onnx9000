"""Provides test_simplifier.py module functionality."""

import pytest
import numpy as np
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.dtypes import DType
from onnx9000.optimize.simplifier.api import simplify
from onnx9000.optimize.simplifier.passes.fusion import fuse_consecutive_transpose
from onnx9000.optimize.simplifier.passes.constant_folding import ConstantFoldingPass
from onnx9000.optimize.simplifier.passes.dce import (
    DCEPass,
    IdentityEliminationPass,
    dead_code_elimination,
)
from onnx9000.optimize.simplifier.passes.validation import detect_cycles


def create_mock_graph():
    """Provides create mock graph functionality and verification."""
    g = Graph("mock")
    n1 = Node("Transpose", ["in1"], ["t1"], {"perm": [1, 0]}, "trans1")
    n2 = Node("Transpose", ["t1"], ["t2"], {"perm": [1, 0]}, "trans2")
    n3 = Node("Relu", ["t2"], ["out1"], {}, "relu1")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    g.inputs = ["in1"]
    g.outputs = ["out1"]
    return g


def test_api_simplify():
    """Tests the test api simplify functionality."""
    g = create_mock_graph()
    g = simplify(g)
    assert len(g.nodes) == 1
    assert g.nodes[0].op_type == "Relu"
    assert g.nodes[0].inputs == ["in1"]


def test_fuse_consecutive_transpose():
    """Tests the test fuse consecutive transpose functionality."""
    g = create_mock_graph()
    dead_code_elimination(g)
    assert len(g.nodes) == 1
    assert g.nodes[0].op_type == "Relu"
    assert g.nodes[0].inputs == ["in1"]


def test_fuse_matmul_add():
    """Tests the test fuse matmul add functionality."""
    g = Graph("mock2")
    g.add_node(Node("MatMul", ["a", "b"], ["c"], {}, "mm"))
    g.add_node(Node("Add", ["c", "d"], ["e"], {}, "add"))
    g.add_node(Node("Relu", ["e"], ["f"], {}, "relu"))
    g.inputs = ["a", "b", "d"]
    g.outputs = ["f"]
    g = simplify(g)
    assert len(g.nodes) == 2
    assert g.nodes[0].op_type == "Gemm"
    assert g.nodes[0].outputs == ["e"]
    assert g.nodes[0].inputs == ["a", "b", "d"]
    assert g.nodes[1].op_type == "Relu"


def test_dead_code_elimination():
    """Tests the test dead code elimination functionality."""
    g = Graph("mock")
    n1 = Node("Add", ["a", "b"], ["c"], {}, "add1")
    n2 = Node("Mul", ["c", "d"], ["e"], {}, "mul1")
    n3 = Node("Sub", ["a", "b"], ["dead"], {}, "dead1")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    g.outputs = ["e"]
    DCEPass().run(g)
    assert len(g.nodes) == 2
    assert "dead" not in [n.outputs[0] for n in g.nodes]


def test_constant_folding_add():
    """Tests the test constant folding add functionality."""
    g = Graph("mock")
    val_a = np.array([1, 2], dtype=np.float32)
    val_b = np.array([3, 4], dtype=np.float32)
    t_a = Tensor("a", shape=(2,), dtype=DType.FLOAT32, data=val_a, is_initializer=True)
    t_b = Tensor("b", shape=(2,), dtype=DType.FLOAT32, data=val_b, is_initializer=True)
    g.add_tensor(t_a)
    g.add_tensor(t_b)
    g.initializers = ["a", "b"]
    g.add_node(Node("Add", ["a", "b"], ["c"], {}, "add1"))
    g.outputs = ["c"]
    ConstantFoldingPass().run(g)
    assert len(g.nodes) == 1
    assert g.nodes[0].op_type == "Constant"
    np.testing.assert_array_equal(
        g.nodes[0].attributes["value"], np.array([4, 6], dtype=np.float32)
    )


def test_constant_folding_partial_identity():
    """Tests the test constant folding partial identity functionality."""
    g = Graph("mock")
    val_b = np.array([0, 0], dtype=np.float32)
    t_b = Tensor("b", shape=(2,), dtype=DType.FLOAT32, data=val_b, is_initializer=True)
    g.add_tensor(t_b)
    g.initializers = ["b"]
    g.add_node(Node("Add", ["a", "b"], ["c"], {}, "add1"))
    g.outputs = ["c"]
    ConstantFoldingPass().run(g)
    assert g.nodes[0].op_type == "Identity"
    assert g.nodes[0].inputs == ["a"]


def test_constant_folding_reshape():
    """Tests the test constant folding reshape functionality."""
    g = Graph("mock")
    val_a = np.array([1, 2, 3, 4], dtype=np.float32)
    val_b = np.array([2, 2], dtype=np.int64)
    g.add_node(Node("Constant", [], ["a"], {"value": val_a}, "const_a"))
    g.add_node(Node("Constant", [], ["b"], {"value": val_b}, "const_b"))
    g.add_node(Node("Reshape", ["a", "b"], ["c"], {}, "reshape"))
    g.outputs = ["c"]
    ConstantFoldingPass().run(g)
    res_nodes = [n for n in g.nodes if n.outputs[0] == "c"]
    assert len(res_nodes) == 1
    assert res_nodes[0].op_type == "Constant"
    np.testing.assert_array_equal(
        res_nodes[0].attributes["value"], np.array([[1, 2], [3, 4]], dtype=np.float32)
    )


def test_constant_folding_mul():
    """Tests the test constant folding mul functionality."""
    g = Graph("mock")
    val_a = np.array([1, 2], dtype=np.float32)
    val_b = np.array([3, 4], dtype=np.float32)
    t_a = Tensor("a", shape=(2,), dtype=DType.FLOAT32, data=val_a, is_initializer=True)
    t_b = Tensor("b", shape=(2,), dtype=DType.FLOAT32, data=val_b, is_initializer=True)
    g.add_tensor(t_a)
    g.add_tensor(t_b)
    g.initializers = ["a", "b"]
    g.add_node(Node("Mul", ["a", "b"], ["c"], {}, "mul1"))
    g.outputs = ["c"]
    ConstantFoldingPass().run(g)
    assert g.nodes[0].op_type == "Constant"
    np.testing.assert_array_equal(
        g.nodes[0].attributes["value"], np.array([3, 8], dtype=np.float32)
    )


def test_constant_folding_slice():
    """Tests the test constant folding slice functionality."""
    g = Graph("mock")
    val_a = np.array([1, 2, 3, 4], dtype=np.float32)
    val_starts = np.array([1], dtype=np.int64)
    val_ends = np.array([3], dtype=np.int64)
    g.add_node(Node("Constant", [], ["a"], {"value": val_a}, "const_a"))
    g.add_node(Node("Constant", [], ["starts"], {"value": val_starts}, "const_s"))
    g.add_node(Node("Constant", [], ["ends"], {"value": val_ends}, "const_e"))
    g.add_node(Node("Slice", ["a", "starts", "ends"], ["c"], {}, "slice"))
    g.outputs = ["c"]
    ConstantFoldingPass().run(g)
    res_nodes = [n for n in g.nodes if n.outputs[0] == "c"]
    assert len(res_nodes) == 1
    assert res_nodes[0].op_type == "Constant"
    np.testing.assert_array_equal(
        res_nodes[0].attributes["value"], np.array([2, 3], dtype=np.float32)
    )


def test_cycle_detection():
    """Tests the test cycle detection functionality."""
    g = Graph("mock")
    g.add_node(Node("Add", ["a", "c"], ["b"], {}, "n1"))
    g.add_node(Node("Add", ["b"], ["c"], {}, "n2"))
    with pytest.raises(RuntimeError):
        detect_cycles(g)


def test_identity_elimination_concat():
    """Tests the test identity elimination concat functionality."""
    g = Graph("mock")
    g.add_node(Node("Concat", ["a"], ["b"], {"axis": 0}, "concat"))
    g.outputs = ["b"]
    IdentityEliminationPass().run(g)
    assert g.outputs[0] == "a"


def test_identity_elimination_cast():
    """Tests the test identity elimination cast functionality."""
    g = Graph("mock")
    g.add_node(Node("Cast", ["a"], ["b"], {"to": 1}, "c1"))
    g.add_node(Node("Cast", ["b"], ["c"], {"to": 1}, "c2"))
    g.outputs = ["c"]
    IdentityEliminationPass().run(g)
    assert g.nodes[1].inputs[0] == "a"
