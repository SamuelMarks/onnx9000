"""TVM submodule for AST and optimization."""

from ..expr import Call, Constant, Expr, Op
from ..visitor import ExprMutator


class AlgebraicSimplifier(ExprMutator):
    """Simplifies algebraic expressions.

    x * 1 -> x
    x * 0 -> 0
    x + 0 -> x
    """

    def is_constant_value(self, expr: Expr, val: float) -> bool:
        """Do the function."""
        if isinstance(expr, Constant):
            # Try to evaluate if a constant is all given val
            # For simplicity, if it's a scalar:
            if isinstance(expr.data, (int, float)) and expr.data == val:
                return True
            # For arrays, we skip for now to avoid numpy dependency
            # It should ideally check the raw memoryview or buffer
        return False

    def visit_call(self, expr: Call) -> Expr:
        """Do the function."""
        new_op = self.visit(expr.op)
        new_args = [self.visit(arg) for arg in expr.args]

        if isinstance(new_op, Op):
            if new_op.name == "Multiply":
                # x * 1 -> x
                if self.is_constant_value(new_args[1], 1.0):
                    return new_args[0]
                if self.is_constant_value(new_args[0], 1.0):
                    return new_args[1]
                # x * 0 -> 0
                if self.is_constant_value(new_args[1], 0.0):
                    return new_args[1]
                if self.is_constant_value(new_args[0], 0.0):
                    return new_args[0]
            elif new_op.name == "Add":
                # x + 0 -> x
                if self.is_constant_value(new_args[1], 0.0):
                    return new_args[0]
                if self.is_constant_value(new_args[0], 0.0):
                    return new_args[1]

        if new_op is not expr.op or any(a is not b for a, b in zip(new_args, expr.args)):
            return Call(op=new_op, args=new_args, attrs=expr.attrs)
        return expr


def simplify_algebra(expr: Expr) -> Expr:
    """Pass to simplify algebraic expressions."""
    return AlgebraicSimplifier().visit(expr)
