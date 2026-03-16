"""Module providing core logic and structural definitions."""

import ast
import inspect
from typing import Any, Callable
from onnx9000.frontend.frontend.builder import GraphBuilder
from onnx9000.frontend.frontend.tensor import Node, Tensor


class ScriptCompiler(ast.NodeVisitor):
    """Class ScriptCompiler implementation."""

    def __init__(self, func: Callable) -> None:
        """Implements the __init__ method."""
        self.func = func
        self.env: dict[str, Any] = {}
        self.builder = GraphBuilder(name=func.__name__)

    def compile(self, *args, **kwargs) -> GraphBuilder:
        """Implements the compile method."""
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

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Implements the visit_FunctionDef method."""
        for stmt in node.body:
            self.visit(stmt)

    def visit_Return(self, node: ast.Return) -> None:
        """Implements the visit_Return method."""
        if node.value:
            val = self.visit(node.value)
            if isinstance(val, Tensor):
                self.builder.outputs.append(val)
            elif isinstance(val, (tuple, list)):
                for v in val:
                    if isinstance(v, Tensor):
                        self.builder.outputs.append(v)
        return None

    def visit_If(self, node: ast.If) -> None:
        """Implements the visit_If method."""
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

    def visit_For(self, node: ast.For) -> None:
        """Implements the visit_For method."""
        self.visit(node.iter)
        for stmt in node.body:
            self.visit(stmt)

    def visit_While(self, node: ast.While) -> None:
        """Implements the visit_While method."""
        self.visit(node.test)
        for stmt in node.body:
            self.visit(stmt)

    def visit_Name(self, node: ast.Name):
        """Implements the visit_Name method."""
        return self.env.get(node.id, None)

    def generic_visit(self, node):
        """Implements the generic_visit method."""
        return super().generic_visit(node)

    def visit_Tuple(self, node: ast.Tuple):
        """Implements the visit_Tuple method."""
        return tuple((self.visit(e) for e in node.elts))

    def visit_List(self, node: ast.List):
        """Implements the visit_List method."""
        return [self.visit(e) for e in node.elts]
