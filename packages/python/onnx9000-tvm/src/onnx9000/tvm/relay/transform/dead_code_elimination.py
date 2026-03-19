"""TVM submodule for AST and optimization."""

from ..expr import Expr, Let, Var
from ..visitor import ExprMutator


class DeadCodeElimination(ExprMutator):
    """Eliminates dead code.

    Specifically, it removes Let bindings where the variable is not used in the body.
    """

    def visit_let(self, expr: Let) -> Expr:
        """Do the function."""
        # First process the body
        new_body = self.visit(expr.body)

        # Check if the bound variable is used in the new body
        from ..analysis import topological_sort

        nodes = topological_sort(new_body)

        # If the variable is not found in any node's children/var, it's unused
        used = False
        for node in nodes:
            if isinstance(node, Var) and node.name_hint == expr.var.name_hint:
                # Basic check - in a real implementation we'd track actual identity
                used = True
                break

        if not used:
            # Skip the Let binding completely, just return the body
            return new_body

        new_var = self.visit(expr.var)
        new_value = self.visit(expr.value)

        if new_var is not expr.var or new_value is not expr.value or new_body is not expr.body:
            return Let(var=new_var, value=new_value, body=new_body)
        return expr


def eliminate_dead_code(expr: Expr) -> Expr:
    """Pass to eliminate dead code."""
    return DeadCodeElimination().visit(expr)
