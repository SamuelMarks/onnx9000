"""Module providing core logic and structural definitions."""

import ast
import inspect
from typing import Callable, Any, Dict, List
from onnx9000.frontends.frontend.builder import GraphBuilder, get_active_builder
from onnx9000.frontends.frontend.tensor import Tensor, Node


class ScriptCompiler(ast.NodeVisitor):
    """Provides semantic functionality and verification."""

    def __init__(self, func: Callable):
        """Provides semantic functionality and verification."""
        self.func = func
        self.env: Dict[str, Any] = {}
        self.builder = GraphBuilder(name=func.__name__)

    def compile(self, *args, **kwargs) -> GraphBuilder:
        """Provides semantic functionality and verification."""
        source = inspect.getsource(self.func)
        import textwrap

        source = textwrap.dedent(source)
        tree = ast.parse(source)
        sig = inspect.signature(self.func)
        for i, param in enumerate(sig.parameters.values()):
            if i < len(args):
                self.env[param.name] = args[i]
            else:
                self.env[param.name] = None
        self.visit(tree)
        return self.builder

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Provides semantic functionality and verification."""
        for stmt in node.body:
            self.visit(stmt)

    def visit_Return(self, node: ast.Return):
        """Provides semantic functionality and verification."""
        if node.value:
            val = self.visit(node.value)
            if isinstance(val, Tensor):
                self.builder.outputs.append(val)
            elif isinstance(val, tuple):
                for v in val:
                    if isinstance(v, Tensor):
                        self.builder.outputs.append(v)
            elif isinstance(val, list):
                for v in val:
                    if isinstance(v, Tensor):
                        self.builder.outputs.append(v)
        return None

    def visit_If(self, node: ast.If):
        """Provides semantic functionality and verification."""
        cond = self.visit(node.test)
        then_builder = GraphBuilder(name=f"{self.builder.name}_then")
        else_builder = GraphBuilder(name=f"{self.builder.name}_else")
        node_ir = Node(
            op_type="If",
            inputs=[cond] if isinstance(cond, Tensor) else [],
            outputs=[],
            attributes={"then_branch": then_builder, "else_branch": else_builder},
        )
        self.builder.add_node(node_ir)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_For(self, node: ast.For):
        """Provides semantic functionality and verification."""
        self.visit(node.iter)
        for stmt in node.body:
            self.visit(stmt)

    def visit_While(self, node: ast.While):
        """Provides semantic functionality and verification."""
        self.visit(node.test)
        for stmt in node.body:
            self.visit(stmt)

    def visit_Name(self, node: ast.Name):
        """Provides semantic functionality and verification."""
        return self.env.get(node.id, None)

    def generic_visit(self, node):
        """Provides semantic functionality and verification."""
        return super().generic_visit(node)

    def visit_Tuple(self, node: ast.Tuple):
        """Provides visit Tuple functionality and verification."""
        return tuple(self.visit(e) for e in node.elts)

    def visit_List(self, node: ast.List):
        """Provides visit List functionality and verification."""
        return [self.visit(e) for e in node.elts]
