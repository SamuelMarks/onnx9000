"""Tests for codegen."""

from onnx9000.c_compiler.codegen import BaseCodegenVisitor, CFamilyCodegen, PythonFamilyCodegen
from onnx9000.core.ir import Graph, Node


def test_base_codegen():
    """Docstring for D103."""
    v = BaseCodegenVisitor()
    assert v.get_var_name() == "v1"


def test_c_family():
    """Docstring for D103."""
    g = Graph("test")
    g.nodes.append(Node(op_type="Add", inputs=[], outputs=[]))
    v = CFamilyCodegen()
    code = v.visit(g)
    assert "#include <stddef.h>" in code
    assert "void forward_test() {" in code
    assert "Tensor v1 = op_add();" in code


def test_python_family():
    """Docstring for D103."""
    g = Graph("test")
    g.nodes.append(Node(op_type="Add", inputs=[], outputs=[]))
    v = PythonFamilyCodegen()
    code = v.visit(g)
    assert "class Model:" in code
    assert "def forward_test(self):" in code
    assert "v1 = add()" in code


def test_base_visit():
    """Docstring for D103."""
    v = BaseCodegenVisitor()
    g = Graph("test")
    g.nodes.append(Node(op_type="Add", inputs=[], outputs=[]))
    assert "NotImplemented Add" in v.visit(g)


def test_imports():
    """Docstring for D103."""
    v = PythonFamilyCodegen()
    v.imports.add("sys")
    g = Graph("test")
    assert "import sys" in v.visit(g)


def test_c_includes():
    """Docstring for D103."""
    v = CFamilyCodegen()
    v.includes.add("<math.h>")
    g = Graph("test")
    assert "#include <math.h>" in v.visit(g)
