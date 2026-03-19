from typing import Callable, List, Set

from .expr import Expr
from .visitor import ExprVisitor


class PostOrderVisitor(ExprVisitor):
    def __init__(self):
        self.visited: set[int] = set()
        self.order: list[Expr] = []

    def visit(self, expr: Expr) -> None:
        expr_id = id(expr)
        if expr_id in self.visited:
            return

        super().visit(expr)

        self.visited.add(expr_id)
        self.order.append(expr)


def post_order_visit(expr: Expr) -> list[Expr]:
    """Returns a list of expressions in post-order (topological sort of dependencies)."""
    visitor = PostOrderVisitor()
    visitor.visit(expr)
    return visitor.order


def topological_sort(expr: Expr) -> list[Expr]:
    """Alias for post_order_visit, providing topological sorting of the AST."""
    return post_order_visit(expr)
