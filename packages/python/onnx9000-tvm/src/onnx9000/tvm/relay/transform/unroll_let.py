from typing import Dict

from ..expr import Expr, Let, Var
from ..visitor import ExprMutator


class LetUnroller(ExprMutator):
    """
    Unrolls let bindings.
    Replaces variables with their bound values in the body.
    """

    def __init__(self):
        self.var_map: dict[str, Expr] = {}

    def visit_var(self, expr: Var) -> Expr:
        if expr.name_hint in self.var_map:
            # Inline the mapped expression
            return self.var_map[expr.name_hint]
        return expr

    def visit_let(self, expr: Let) -> Expr:
        # Evaluate the value first
        new_value = self.visit(expr.value)

        # Add to the mapping
        old_val = self.var_map.get(expr.var.name_hint)
        self.var_map[expr.var.name_hint] = new_value

        # Evaluate the body with the new mapping
        new_body = self.visit(expr.body)

        # Restore old mapping to respect scope
        if old_val is not None:
            self.var_map[expr.var.name_hint] = old_val
        else:
            del self.var_map[expr.var.name_hint]

        return new_body


def unroll_let(expr: Expr) -> Expr:
    """Pass to unroll let bindings."""
    return LetUnroller().visit(expr)
