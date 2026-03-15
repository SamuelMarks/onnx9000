"""
Provides the AST parser that translates Python functions into ONNX graphs.
Handles mapping of variables, loops, conditionals, and node generation.
"""

import ast
import inspect
import textwrap
from typing import Any, Callable, Dict, List

from onnx9000.core.dtypes import DType
from onnx9000.script.builder import GraphBuilder
from onnx9000.script.op import op
from onnx9000.script.var import Var


class ScriptParser(ast.NodeVisitor):
    """Parses a Python function AST into an ONNX GraphBuilder."""

    def __init__(self, globals_dict: Dict[str, Any]):
        """Initializes the parser with a dictionary of global variables."""
        self.builder = GraphBuilder()
        self.globals_dict = globals_dict
        self.locals_dict: Dict[str, Var] = {}

    def parse(self, func: Callable) -> GraphBuilder:
        """Parses a given Python function and populates the GraphBuilder."""
        source = inspect.getsource(func)
        source = textwrap.dedent(source)
        tree = ast.parse(source)

        func_def = tree.body[0]
        if not isinstance(func_def, ast.FunctionDef):
            raise ValueError("Expected a function definition")

        self.builder.name = func_def.name

        # Parse arguments as inputs
        for arg in func_def.args.args:
            # We default to float32 unknown shape if not hinted, or we can use typing
            # In a real system we'd parse the type hint annotations
            shape: tuple[Any, ...] = ("?",)
            dtype = DType.FLOAT32

            # Simple annotation parsing
            if arg.annotation and isinstance(arg.annotation, ast.Subscript):
                # E.g. Float[10, 20]
                dtype = DType.FLOAT32

            var = self.builder.add_input(arg.arg, dtype, shape)
            self.locals_dict[arg.arg] = var

        # Parse body
        for stmt in func_def.body:
            self.visit(stmt)

        return self.builder

    def visit(self, node: ast.AST) -> Any:
        """Visits an AST node, wrapping exceptions with line number details."""
        try:
            return super().visit(node)
        except Exception as e:
            if hasattr(node, "lineno") and not str(e).startswith("Parse error"):
                raise ValueError(f"Parse error at line {node.lineno}: {str(e)}") from e
            raise

    def visit_Expr(self, node: ast.Expr) -> Any:
        """Translates expression statements, ignoring docstrings."""
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.visit(node.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Translates Python assignment statements into ONNX variable bindings."""
        val = self.visit(node.value)
        if len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                if isinstance(val, Var):
                    val.rename(target.id)
                self.locals_dict[target.id] = val
            elif isinstance(target, ast.Tuple):
                # Multiple assignment (e.g. a, b = op.Split(...))
                if isinstance(val, tuple):
                    for t, v in zip(target.elts, val):
                        if isinstance(t, ast.Name):
                            v.rename(t.id)
                            self.locals_dict[t.id] = v

    def visit_Call(self, node: ast.Call) -> Any:
        """Translates function calls into ONNX node invocations or subgraph inlining."""
        args = [self.visit(arg) for arg in node.args]
        kwargs = {
            kw.arg: self.visit(kw.value) for kw in node.keywords if kw.arg is not None
        }

        # e.g. op.Add(X, Z)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "op"
        ):
            op_name = node.func.attr
            op_func = getattr(op, op_name)
            with self.builder:
                return op_func(*args, **kwargs)

        # Call to a local/global function
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name:
            # Check if it's a @script decorated function
            # Since @script wrapper has _is_onnx_script
            func_obj = self.globals_dict.get(func_name) or self.locals_dict.get(
                func_name
            )
            if hasattr(func_obj, "_is_onnx_script"):
                # Inline the subgraph by getting its builder
                sub_builder = func_obj.to_builder()  # type: ignore
                # Rename nodes to avoid conflicts
                sub_builder.rename_all(f"inline_{func_name}_{id(node)}")

                # Replace sub_builder inputs with our passed arguments
                for i, arg_var in enumerate(args):
                    sub_in = sub_builder.inputs[i]["name"]
                    # We need to map `sub_in` to `arg_var.name`
                    # Actually, the sub_builder nodes just need to read `arg_var.name` instead of `sub_in`.
                    for n in sub_builder.nodes:
                        n.inputs = [
                            arg_var.name if x == sub_in else x for x in n.inputs
                        ]

                # Merge the nodes and initializers
                self.builder.merge(sub_builder)

                # Return the mapped outputs
                out_vars = [Var(name=out["name"]) for out in sub_builder.outputs]
                if len(out_vars) == 1:
                    return out_vars[0]
                return tuple(out_vars)

        raise ValueError(f"Unsupported call: {ast.dump(node)}")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        """Translates binary operations into their corresponding ONNX mathematical nodes."""
        left = self.visit(node.left)
        right = self.visit(node.right)

        with self.builder:
            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                return left / right
            elif isinstance(node.op, ast.MatMult):
                return left @ right
            elif isinstance(node.op, ast.Pow):
                return left**right
        raise ValueError(f"Unsupported binary operator: {type(node.op)}")

    def visit_Compare(self, node: ast.Compare) -> Any:
        """Translates comparison operations into ONNX logical nodes."""
        if len(node.ops) > 1 or len(node.comparators) > 1:
            raise ValueError("Multiple comparisons not supported")
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        op_node = node.ops[0]
        with self.builder:
            if isinstance(op_node, ast.Gt):
                return left > right
            elif isinstance(op_node, ast.Lt):
                return left < right
            elif isinstance(op_node, ast.Eq):
                return left == right
            elif isinstance(op_node, ast.NotEq):
                return left != right
        raise ValueError(f"Unsupported comparison: {type(op_node)}")

    def visit_For(self, node: ast.For) -> None:
        """Translates Python for-loops into ONNX Loop operations."""
        max_trip_count = self.visit(node.iter)

        parent_builder = self.builder
        parent_locals = self.locals_dict.copy()

        body_builder = GraphBuilder(name=f"{parent_builder.name}_for_body")
        self.builder = body_builder
        self.locals_dict = parent_locals.copy()

        iter_num = self.builder.add_input("iteration_num", DType.INT64, (1,))
        if isinstance(node.target, ast.Name):
            self.locals_dict[node.target.id] = iter_num

        cond_in = self.builder.add_input("cond_in", DType.BOOL, (1,))

        for stmt in node.body:
            self.visit(stmt)

        body_locals = self.locals_dict.copy()
        self.builder = parent_builder
        self.locals_dict = parent_locals

        carried_keys = sorted(
            k
            for k, v in body_locals.items()
            if k in parent_locals and v is not parent_locals[k]
        )

        with body_builder:
            import numpy as np

            true_cond = op.Constant(np.array([True], dtype=np.bool_))
            body_builder.add_output(true_cond)

        for k in carried_keys:
            in_var = body_builder.add_input(f"in_{k}", DType.FLOAT32, ("?",))
            old_var = parent_locals[k]
            for n in body_builder.nodes:
                body_builder.replace_input(n, old_var, in_var)
            body_builder.add_output(body_locals[k])

        num_outputs = len(carried_keys)
        with self.builder:
            import numpy as np

            cond = op.Constant(np.array([True], dtype=np.bool_))
            out_vars = op.Loop(
                max_trip_count, cond, body_builder, num_outputs=num_outputs
            )

        if num_outputs == 1:
            self.locals_dict[carried_keys[0]] = out_vars  # type: ignore
        elif num_outputs > 1:
            for i, k in enumerate(carried_keys):
                self.locals_dict[k] = out_vars[i]  # type: ignore

    def visit_While(self, node: ast.While) -> None:
        """Translates Python while-loops into ONNX Loop operations."""
        parent_builder = self.builder
        parent_locals = self.locals_dict.copy()

        body_builder = GraphBuilder(name=f"{parent_builder.name}_while_body")
        self.builder = body_builder
        self.locals_dict = parent_locals.copy()

        iter_num = self.builder.add_input("iteration_num", DType.INT64, (1,))
        cond_in = self.builder.add_input("cond_in", DType.BOOL, (1,))

        for stmt in node.body:
            self.visit(stmt)

        cond_out = self.visit(node.test)

        body_locals = self.locals_dict.copy()
        self.builder = parent_builder
        self.locals_dict = parent_locals

        carried_keys = sorted(
            k
            for k, v in body_locals.items()
            if k in parent_locals and v is not parent_locals[k]
        )

        body_builder.add_output(cond_out)
        for k in carried_keys:
            in_var = body_builder.add_input(f"in_{k}", DType.FLOAT32, ("?",))
            old_var = parent_locals[k]
            for n in body_builder.nodes:
                body_builder.replace_input(n, old_var, in_var)
            body_builder.add_output(body_locals[k])

        num_outputs = len(carried_keys)
        with self.builder:
            import numpy as np

            max_trip = op.Constant(np.array([9223372036854775807], dtype=np.int64))
            cond = self.visit(node.test)
            out_vars = op.Loop(max_trip, cond, body_builder, num_outputs=num_outputs)

        if num_outputs == 1:
            self.locals_dict[carried_keys[0]] = out_vars  # type: ignore
        elif num_outputs > 1:
            for i, k in enumerate(carried_keys):
                self.locals_dict[k] = out_vars[i]  # type: ignore

    def visit_ListComp(self, node: ast.ListComp) -> Any:
        """Raises an error indicating list comprehensions are not supported."""
        raise ValueError(
            "List comprehensions cannot be mapped to ONNX directly. Unroll statically."
        )

    def visit_Name(self, node: ast.Name) -> Any:
        """Resolves variable names to ONNX variables or captured global constants."""
        if node.id in self.locals_dict:
            return self.locals_dict[node.id]
        elif node.id in self.globals_dict:
            # Capture closed-over variables
            val = self.globals_dict[node.id]
            with self.builder:
                var = op.Constant(val)
                self.locals_dict[node.id] = var
                return var
        raise ValueError(f"Unknown variable: {node.id}")

    def visit_Constant(self, node: ast.Constant) -> Any:
        """Translates Python constants into ONNX Constant nodes."""
        with self.builder:
            return op.Constant(node.value)

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        """Translates Python tuples into tuples of ONNX variables."""
        return tuple(self.visit(el) for el in node.elts)

    def visit_Return(self, node: ast.Return) -> None:
        """Translates return statements into ONNX graph outputs."""
        if node.value is None:
            return
        val = self.visit(node.value)
        if isinstance(val, tuple):
            for i, v in enumerate(val):
                self.builder.add_output(v, name=f"output_{i}")
        else:
            self.builder.add_output(val, name="output_0")

    def visit_If(self, node: ast.If) -> None:
        """Translates Python if-statements into ONNX If operations."""
        cond = self.visit(node.test)

        parent_builder = self.builder
        parent_locals = self.locals_dict.copy()

        # Then branch
        then_builder = GraphBuilder(name=f"{parent_builder.name}_then")
        self.builder = then_builder
        self.locals_dict = parent_locals.copy()
        for stmt in node.body:
            self.visit(stmt)
        then_locals = self.locals_dict.copy()

        # Else branch
        else_builder = GraphBuilder(name=f"{parent_builder.name}_else")
        self.builder = else_builder
        self.locals_dict = parent_locals.copy()
        for stmt in node.orelse:
            self.visit(stmt)
        else_locals = self.locals_dict.copy()

        self.builder = parent_builder
        self.locals_dict = parent_locals

        # Determine outputs based on changed variables
        output_keys = []
        # Maintain order by sorting keys
        all_keys = sorted(list(set(then_locals.keys()).union(else_locals.keys())))
        for k in all_keys:
            t_var = then_locals.get(k)
            e_var = else_locals.get(k)
            p_var = parent_locals.get(k)
            if t_var is not p_var or e_var is not p_var:
                if t_var is None or e_var is None:
                    raise ValueError(
                        f"Variable {k} must be defined in both branches of If."
                    )
                output_keys.append(k)
                then_builder.add_output(t_var)
                else_builder.add_output(e_var)

        num_outputs = len(output_keys)
        with self.builder:
            out_vars = op.If(
                cond,
                then_branch=then_builder,
                else_branch=else_builder,
                num_outputs=num_outputs,
            )

        if num_outputs == 1:
            self.locals_dict[output_keys[0]] = out_vars  # type: ignore
            out_vars.rename(output_keys[0])  # type: ignore
        elif num_outputs > 1:
            for i, k in enumerate(output_keys):
                self.locals_dict[k] = out_vars[i]  # type: ignore
                out_vars[i].rename(k)  # type: ignore


def script(func: Callable) -> Callable:
    """Decorator to convert a Python function to an ONNX GraphBuilder/Model."""
    # Capture globals at definition time
    frame = inspect.currentframe()
    if frame is not None and frame.f_back is not None:
        globals_dict = frame.f_back.f_globals.copy()
        globals_dict.update(frame.f_back.f_locals)
    else:
        globals_dict = {}

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Executes the parsed ONNX graph by building and running it."""
        parser = ScriptParser(globals_dict)
        builder = parser.parse(func)
        return builder.build()

    wrapper._is_onnx_script = True  # type: ignore
    wrapper.to_builder = lambda: ScriptParser(globals_dict).parse(func)  # type: ignore
    return wrapper
