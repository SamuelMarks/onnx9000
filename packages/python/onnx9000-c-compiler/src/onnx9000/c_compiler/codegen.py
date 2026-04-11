"""Codegen family visitors."""

import logging
from typing import Any

from onnx9000.core.ir import Graph, Node


class BaseCodegenVisitor:
    """Handles topological sorting, variable naming, and basic memory lifecycle tracking."""

    def __init__(self):
        """Initialize the base visitor."""
        self.var_count = 0
        self.env = {}

    def get_var_name(self, prefix: str = "v") -> str:
        """Get a fresh variable name."""
        self.var_count += 1
        return f"{prefix}{self.var_count}"

    def visit(self, graph: Graph) -> str:
        """Visit graph nodes and generate code."""
        # Sort is assumed already done by Graph builder/IR
        code = []
        for node in graph.nodes:
            code.append(self.visit_node(node))
        return "\n".join(code)

    def visit_node(self, node: Node) -> str:
        """Visit a specific node."""
        return f"/* Unknown operation: {node.op_type} */"


class CFamilyCodegen(BaseCodegenVisitor):
    """Adds bracket {} scoping, type declarations, and #include management."""

    def __init__(self):
        """Initialize the CFamily visitor."""
        super().__init__()
        self.includes = {"<stddef.h>", "<stdint.h>"}

    def visit_node(self, node: Node) -> str:
        """Visit a node to generate C statement."""
        # Mock node generation
        out_var = self.get_var_name()
        return f"    Tensor {out_var} = op_{node.op_type.lower()}();"

    def visit(self, graph: Graph) -> str:
        """Visit the graph and generate a C function."""
        code = []
        for inc in sorted(self.includes):
            code.append(f"#include {inc}")
        code.append("")
        code.append(f"void forward_{graph.name}() {{")
        for node in graph.nodes:
            code.append(self.visit_node(node))
        code.append("}")
        return "\n".join(code)


class PythonFamilyCodegen(BaseCodegenVisitor):
    """Adds indentation management, import handling. Reused by PyTorch, Flax, and Keras exporters."""

    def __init__(self):
        """Initialize the PythonFamily visitor."""
        super().__init__()
        self.imports = set()

    def visit_node(self, node: Node) -> str:
        """Visit a node to generate Python statement."""
        out_var = self.get_var_name()
        return f"        {out_var} = {node.op_type.lower()}()"

    def visit(self, graph: Graph) -> str:
        """Visit the graph and generate a Python class."""
        code = []
        for imp in sorted(self.imports):
            code.append(f"import {imp}")
        code.append("")
        code.append("class Model:")
        code.append(f"    def forward_{graph.name}(self):")
        for node in graph.nodes:
            code.append(self.visit_node(node))
        code.append("        pass")
        return "\n".join(code)
