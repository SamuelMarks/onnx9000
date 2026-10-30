"""Unit tests for C, Python, and base code generation visitors."""

from onnx9000.c_compiler.codegen import (
    BaseCodegenVisitor,
    CFamilyCodegen,
    PythonFamilyCodegen,
)
from onnx9000.core.ir import Graph, Node


def test_codegen() -> None:
    """Verify BaseCodegenVisitor, CFamilyCodegen, and PythonFamilyCodegen output generation."""
    graph = Graph("test")
    node1 = Node("Add", ["A", "B"], ["C"])
    node2 = Node("Mul", ["C", "D"], ["E"])
    graph.nodes = [node1, node2]

    base = BaseCodegenVisitor()
    out = base.visit(graph)
    assert "Unknown operation: Add" in out

    c = CFamilyCodegen()
    out_c = c.visit(graph)
    assert "void forward_test()" in out_c
    assert "op_add" in out_c

    p = PythonFamilyCodegen()
    out_p = p.visit(graph)
    assert "class Model:" in out_p
    assert "add()" in out_p
