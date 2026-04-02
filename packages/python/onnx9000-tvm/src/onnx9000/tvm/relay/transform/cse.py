"""TVM submodule for AST and optimization."""

from ..expr import Call, Constant, Expr, Op, TupleExpr, TupleGetItem, Var
from ..visitor import ExprMutator


class CommonSubexprEliminator(ExprMutator):
    """Core class for TVM AST node or pass."""

    def __init__(self):
        """Magic method."""
        """Initialize."""
        """Do the function."""
        self.expr_map: dict[tuple, Expr] = {}

    def hash_expr(self, expr: Expr) -> tuple:
        """Do the function."""
        if isinstance(expr, Var):
            return ("Var", expr.name_hint)
        elif isinstance(expr, Constant):
            # For data, we assume it's hashable or we hash its memory address
            # Realistically, hashing large arrays requires specialized logic
            try:
                data_hash = hash(expr.data)
            except TypeError:
                data_hash = id(expr.data)
            return ("Constant", data_hash)
        elif isinstance(expr, Op):
            return ("Op", expr.name)
        elif isinstance(expr, Call):
            op_hash = self.hash_expr(expr.op)
            arg_hashes = tuple(self.hash_expr(arg) for arg in expr.args)
            attr_hash = tuple(sorted(expr.attrs.items())) if expr.attrs else ()
            return ("Call", op_hash, arg_hashes, attr_hash)
        elif isinstance(expr, TupleExpr):
            field_hashes = tuple(self.hash_expr(f) for f in expr.fields)
            return ("TupleExpr", field_hashes)
        elif isinstance(expr, TupleGetItem):
            val_hash = self.hash_expr(expr.tuple_value)
            return ("TupleGetItem", val_hash, expr.index)
        else:
            return ("Unknown", id(expr))

    def visit(self, expr: Expr) -> Expr:
        """Do the function."""
        # First check if we've already computed an identical expression
        h = self.hash_expr(expr)
        if h in self.expr_map:
            return self.expr_map[h]

        # If not, mutate children
        new_expr = super().visit(expr)

        # After mutation, the hash might have changed
        new_h = self.hash_expr(new_expr)
        if new_h in self.expr_map:
            return self.expr_map[new_h]

        self.expr_map[new_h] = new_expr
        return new_expr


def eliminate_common_subexpr(expr: Expr) -> Expr:
    """Pass to perform common subexpression elimination."""
    return CommonSubexprEliminator().visit(expr)
