"""TVM submodule for AST and optimization."""

from typing import Callable

from ..expr import Call, Constant, Expr, Op
from ..visitor import ExprMutator


class ConstantFolder(ExprMutator):
    """Folds operations applied to constants.

    Requires a registry of evaluator functions for operations.
    """

    def __init__(self, evaluators: dict[str, Callable]):
        """Magic method."""
        """Initialize."""
        """Do the function."""
        self.evaluators = evaluators

    def visit_call(self, expr: Call) -> Expr:
        """Do the function."""
        new_op = self.visit(expr.op)
        new_args = [self.visit(arg) for arg in expr.args]

        # Check if all args are constants
        if all(isinstance(arg, Constant) for arg in new_args) and isinstance(new_op, Op):
            op_name = new_op.name
            if op_name in self.evaluators:
                evaluator = self.evaluators[op_name]
                # Evaluate the constant expression
                const_args = [arg.data for arg in new_args]
                result_data = evaluator(*const_args, **(expr.attrs or {}))
                return Constant(data=result_data)

        if new_op is not expr.op or any(a is not b for a, b in zip(new_args, expr.args)):
            return Call(op=new_op, args=new_args, attrs=expr.attrs)
        return expr


def fold_constant(expr: Expr, evaluators: dict[str, Callable] = None) -> Expr:
    """Pass to fold constant expressions."""
    if evaluators is None:
        evaluators = {}
    return ConstantFolder(evaluators).visit(expr)
