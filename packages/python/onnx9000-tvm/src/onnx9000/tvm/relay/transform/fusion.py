"""TVM submodule for AST and optimization."""

from ..expr import Call, Constant, Expr, Function, Op, Var
from ..visitor import ExprMutator


class OpFusionDetector(ExprMutator):
    """Detects fusable subgraphs and converts them into fused functions.

    Example: Conv + ReLU -> FusedConvReLU.
    """

    def __init__(self):
        """Magic method."""
        """Initialize."""
        """Do the function."""
        # We define simple fusable patterns.
        # This maps an operation name to a list of ops it can fuse with (if they are consumers).
        self.fusable_rules = {"Conv": ["Relu", "Add"], "MatMul": ["Add"], "Add": ["Relu"]}
        self.fused_count = 0

    def visit_call(self, expr: Call) -> Expr:
        """Do the function."""
        new_op = self.visit(expr.op)
        new_args = [self.visit(arg) for arg in expr.args]

        # Check if we are a target for fusion
        # A simple bottom-up greedy fusion: if a child is a call to `Conv` and we are `Relu`
        if isinstance(new_op, Op) and new_op.name in ["Relu", "Add"]:
            # Check the first argument
            if isinstance(new_args[0], Call) and isinstance(new_args[0].op, Op):
                child_op_name = new_args[0].op.name
                if (
                    child_op_name in self.fusable_rules
                    and new_op.name in self.fusable_rules[child_op_name]
                ):
                    # We can fuse!
                    # Create a composite function. For simplicity, we create an Op named Fused_<op1>_<op2>
                    # In true TVM, we'd extract a Function and mark it with Primitive=1
                    fused_name = f"Fused_{child_op_name}_{new_op.name}"

                    # Gather all inputs to the fused subgraph.
                    # The first arg's args + the rest of our args
                    fused_args = new_args[0].args + new_args[1:]

                    # Create new fused op call
                    fused_call = Call(op=Op(fused_name), args=fused_args, attrs={"fused": True})
                    return fused_call

        if new_op is not expr.op or any(a is not b for a, b in zip(new_args, expr.args)):
            return Call(op=new_op, args=new_args, attrs=expr.attrs)
        return expr


def fuse_ops(expr: Expr) -> Expr:
    """Pass to fuse operators."""
    return OpFusionDetector().visit(expr)
