from typing import Any

from .expr import Call, Constant, Expr, Function, If, Let, Op, TupleExpr, TupleGetItem, Var
from .visitor import ExprVisitor


class Printer(ExprVisitor):
    def __init__(self):
        self.indent = 0
        self.result = ""
        self.var_count = 0

    def get_var_name(self, var: Var) -> str:
        return f"%{var.name_hint}"

    def visit_var(self, expr: Var) -> Any:
        return self.get_var_name(expr)

    def visit_constant(self, expr: Constant) -> Any:
        return f"meta[Constant][{id(expr)}]"

    def visit_op(self, expr: Op) -> Any:
        return f"op({expr.name})"

    def visit_call(self, expr: Call) -> Any:
        op_str = self.visit(expr.op)
        args_str = ", ".join(str(self.visit(arg)) for arg in expr.args)
        attrs_str = ""
        if expr.attrs:
            attrs_str = ", " + ", ".join(f"{k}={v}" for k, v in expr.attrs.items())
        return f"{op_str}({args_str}{attrs_str})"

    def visit_tuple(self, expr: TupleExpr) -> Any:
        fields_str = ", ".join(str(self.visit(f)) for f in expr.fields)
        return f"({fields_str})"

    def visit_tuple_getitem(self, expr: TupleGetItem) -> Any:
        t_str = self.visit(expr.tuple_value)
        return f"{t_str}.{expr.index}"

    def visit_let(self, expr: Let) -> Any:
        var_str = self.visit(expr.var)
        val_str = self.visit(expr.value)

        self.result += f"{'  ' * self.indent}let {var_str} = {val_str};\n"

        body_str = self.visit(expr.body)
        if isinstance(expr.body, (Let, Function, If)):
            return body_str
        else:
            self.result += f"{'  ' * self.indent}{body_str}\n"
            return ""

    def visit_if(self, expr: If) -> Any:
        cond_str = self.visit(expr.cond)
        res = f"if ({cond_str}) {{\n"
        self.indent += 1
        true_str = self.visit(expr.true_branch)
        if not isinstance(expr.true_branch, (Let, If, Function)):
            res += f"{'  ' * self.indent}{true_str}\n"
        res += f"{'  ' * (self.indent - 1)}}} else {{\n"
        false_str = self.visit(expr.false_branch)
        if not isinstance(expr.false_branch, (Let, If, Function)):
            res += f"{'  ' * self.indent}{false_str}\n"
        self.indent -= 1
        res += f"{'  ' * self.indent}}}"
        return res

    def visit_function(self, expr: Function) -> Any:
        params_str = ", ".join(self.visit(p) for p in expr.params)
        res = f"fn ({params_str}) {{\n"
        self.indent += 1
        body_str = self.visit(expr.body)
        if not isinstance(expr.body, (Let, If, Function)):
            res += f"{'  ' * self.indent}{body_str}\n"
        self.indent -= 1
        res += f"{'  ' * self.indent}}}"
        return res


def astext(expr: Expr) -> str:
    """Returns a textual representation of the IR."""
    printer = Printer()
    res = printer.visit(expr)
    if isinstance(expr, (Let, If, Function)):
        return printer.result + res
    return res
