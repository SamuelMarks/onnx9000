"""Implementation of ast parser (converters / frontend / ast_parser.py) providing ScriptCompiler."""

import ast
import inspect
from collections.abc import Callable

from onnx9000.converters.frontend.builder import GraphBuilder
from onnx9000.converters.frontend.tensor import Node, Tensor


class ScriptCompiler(ast.NodeVisitor):
    """Class ScriptCompiler implementation."""

    def __init__(self, func: Callable) -> None:
        """Initialize ScriptCompiler with configured parameters and state.

        Args:
            func: Parameter specifying func (Callable).
        """
        self.func = func
        self.env: dict[str, int | float | str | bool | bytes | list | dict | tuple | None] = {}
        self.builder = GraphBuilder(name=func.__name__)

    def compile(
        self,
        *args: int | float | str | bool | bytes | list | dict | tuple | None,
        **kwargs: int | float | str | bool | bytes | list | dict | tuple | None,
    ) -> GraphBuilder:
        """Execute compile on ScriptCompiler.

        Args:
            *args: Variadic positional arguments for args.
            **kwargs: Variadic keyword arguments for kwargs.

        Returns:
            Result of compile (GraphBuilder).
        """
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
        """Execute visit function def on ScriptCompiler.

        Args:
            node: Parameter specifying node (ast.FunctionDef).
        """
        for stmt in node.body:
            self.visit(stmt)

    def visit_Return(self, node: ast.Return) -> None:
        """Execute visit return on ScriptCompiler.

        Args:
            node: Parameter specifying node (ast.Return).
        """
        if node.value:
            val = self.visit(node.value)
            if isinstance(val, Tensor):
                self.builder.outputs.append(val)
            elif isinstance(val, (tuple, list)):
                for v in val:
                    if isinstance(v, Tensor):
                        self.builder.outputs.append(v)

    def visit_If(self, node: ast.If) -> None:
        """Execute visit if on ScriptCompiler.

        Args:
            node: Parameter specifying node (ast.If).
        """
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
        """Execute visit for on ScriptCompiler.

        Args:
            node: Parameter specifying node (ast.For).
        """
        self.visit(node.iter)
        for stmt in node.body:
            self.visit(stmt)

    def visit_While(self, node: ast.While) -> None:
        """Execute visit while on ScriptCompiler.

        Args:
            node: Parameter specifying node (ast.While).
        """
        self.visit(node.test)
        for stmt in node.body:
            self.visit(stmt)

    def visit_Name(
        self, node: ast.Name
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute visit name on ScriptCompiler.

        Args:
            node: Parameter specifying node (ast.Name).

        Returns:
            Result of visit name.
        """
        return self.env.get(node.id, None)

    def generic_visit(
        self, node: int | float | str | bool | bytes | list | dict | tuple | None
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute generic visit on ScriptCompiler.

        Args:
            node: Parameter specifying node.

        Returns:
            Result of generic visit.
        """
        return super().generic_visit(node)

    def visit_Tuple(
        self, node: ast.Tuple
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute visit tuple on ScriptCompiler.

        Args:
            node: Parameter specifying node (ast.Tuple).

        Returns:
            Result of visit tuple.
        """
        return tuple(self.visit(e) for e in node.elts)

    def visit_List(
        self, node: ast.List
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Execute visit list on ScriptCompiler.

        Args:
            node: Parameter specifying node (ast.List).

        Returns:
            Result of visit list.
        """
        return [self.visit(e) for e in node.elts]
