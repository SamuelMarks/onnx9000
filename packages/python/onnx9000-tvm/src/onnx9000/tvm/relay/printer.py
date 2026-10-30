"""Relay IR text printer."""

from __future__ import annotations

from .expr import Call, Constant, Expr, Function, If, Let, Op, TupleExpr, TupleGetItem, Var
from .visitor import ExprVisitor

NL = chr(10)


class Printer(ExprVisitor):
    """Text printer visitor for Relay expressions."""

    def __init__(self) -> None:
        """Initialize the printer state."""
        self.indent = 0
        self.result = ""
        self.var_count = 0

    def get_var_name(self, var: Var) -> str:
        """Format a variable name with the '%' prefix."""
        return f"%{var.name_hint}"

    def visit_var(self, expr: Var) -> str:
        """Format a Var node."""
        return self.get_var_name(expr)

    def visit_constant(self, expr: Constant) -> str:
        """Format a Constant node."""
        return f"meta[Constant][{id(expr)}]"

    def visit_op(self, expr: Op) -> str:
        """Format an Op node."""
        return f"op({expr.name})"

    def visit_call(self, expr: Call) -> str:
        """Format a Call node."""
        op_str = self.visit(expr.op)
        args_str = ", ".join(str(self.visit(arg)) for arg in expr.args)
        attrs_str = ""
        if expr.attrs:
            attrs_str = ", " + ", ".join(f"{k}={v}" for k, v in expr.attrs.items())
        return f"{op_str}({args_str}{attrs_str})"

    def visit_tuple(self, expr: TupleExpr) -> str:
        """Format a TupleExpr node."""
        fields_str = ", ".join(str(self.visit(f)) for f in expr.fields)
        return f"({fields_str})"

    def visit_tuple_getitem(self, expr: TupleGetItem) -> str:
        """Format a TupleGetItem node."""
        t_str = self.visit(expr.tuple_value)
        return f"{t_str}.{expr.index}"

    def visit_let(self, expr: Let) -> str:
        """Format a Let node."""
        var_str = self.visit(expr.var)
        val_str = self.visit(expr.value)

        self.result += f"{'  ' * self.indent}let {var_str} = {val_str};" + NL

        body_str = str(self.visit(expr.body))
        if isinstance(expr.body, (Let, Function, If)):
            return body_str
        self.result += f"{'  ' * self.indent}{body_str}" + NL
        return ""

    def visit_if(self, expr: If) -> str:
        """Format an If node."""
        cond_str = self.visit(expr.cond)
        res = f"if ({cond_str}) {{" + NL
        self.indent += 1
        true_str = self.visit(expr.true_branch)
        if not isinstance(expr.true_branch, (Let, If, Function)):
            res += f"{'  ' * self.indent}{true_str}" + NL
        res += f"{'  ' * (self.indent - 1)}}} else {{" + NL
        false_str = self.visit(expr.false_branch)
        if not isinstance(expr.false_branch, (Let, If, Function)):
            res += f"{'  ' * self.indent}{false_str}" + NL
        self.indent -= 1
        res += f"{'  ' * self.indent}}}"
        return res

    def visit_function(self, expr: Function) -> str:
        """Format a Function node."""
        params_str = ", ".join(str(self.visit(p)) for p in expr.params)
        res = f"fn ({params_str}) {{" + NL
        self.indent += 1
        body_str = self.visit(expr.body)
        if not isinstance(expr.body, (Let, If, Function)):
            res += f"{'  ' * self.indent}{body_str}" + NL
        self.indent -= 1
        res += f"{'  ' * self.indent}}}"
        return res


def astext(expr: Expr) -> str:
    """Return a textual representation of the IR."""
    printer = Printer()
    res = str(printer.visit(expr))
    if isinstance(expr, (Let, If, Function)):
        return printer.result + res
    return res
