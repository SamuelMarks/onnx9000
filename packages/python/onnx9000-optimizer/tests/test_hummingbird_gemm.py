"""Tests the hummingbird gemm module functionality."""

from onnx9000.core.ir import Graph
from onnx9000.optimizer.hummingbird.gemm import (
    GemmCompiler,
    compile_boosting_gemm,
    compile_forest_gemm,
)
from onnx9000.optimizer.hummingbird.memory import TreeAbstractions


def test_gemm_compiler() -> None:
    """Tests the gemm compiler functionality."""
    g = Graph(name="test_gemm")
    tree = TreeAbstractions()
    tree.add_node(0, 1.5, 1, 2, 0.0)
    tree.add_node(1, 0.0, -1, -1, 10.0)
    tree.add_node(1, 0.0, -1, -1, 20.0)

    compiler = GemmCompiler(tree)
    compiler.compile(g)

    # Check that tensors are created
    assert "test_gemm_gemm_a" in g.tensors
    assert "test_gemm_gemm_b" in g.tensors
    assert "test_gemm_gemm_c" in g.tensors
    assert "test_gemm_gemm_d" in g.tensors

    # Check nodes
    op_types = [node.op_type for node in g.nodes]
    assert "MatMul" in op_types
    assert "Less" in op_types
    assert "Sign" in op_types
    assert "Relu" in op_types
    assert "ArgMax" in op_types


def test_forest_gemm() -> None:
    """Tests the forest gemm functionality."""
    g = Graph(name="test_forest")
    compile_forest_gemm(g, [TreeAbstractions(), TreeAbstractions()])
    # It passes for now


def test_boosting_gemm() -> None:
    """Tests the boosting gemm functionality."""
    g = Graph(name="test_boost")
    compile_boosting_gemm(g, [TreeAbstractions(), TreeAbstractions()])
    # It passes for now
