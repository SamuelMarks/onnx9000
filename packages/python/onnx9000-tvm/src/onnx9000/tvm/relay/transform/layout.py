"""TVM submodule for AST and optimization."""

from ..expr import Call, Expr, Op
from ..visitor import ExprMutator


class LayoutTransform(ExprMutator):
    """Transforms layout of operators (e.g. NCHW to NHWC)."""

    def __init__(self, src_layout: str = "NCHW", dst_layout: str = "NHWC"):
        """Magic method."""
        """Initialize."""
        """Do the function."""
        self.src_layout = src_layout
        self.dst_layout = dst_layout

    def _create_transpose(self, expr: Expr, perm: list) -> Expr:
        """Do the function."""
        return Call(op=Op("Transpose"), args=[expr], attrs={"perm": perm})

    def visit_call(self, expr: Call) -> Expr:
        """Do the function."""
        new_op = self.visit(expr.op)
        new_args = [self.visit(arg) for arg in expr.args]

        if isinstance(new_op, Op):
            # Example: converting a Conv2D
            if new_op.name == "Conv" and self.src_layout == "NCHW" and self.dst_layout == "NHWC":
                # Assuming attrs contains original layout
                layout = expr.attrs.get("layout", "NCHW") if expr.attrs else "NCHW"
                if layout == "NCHW":
                    # We need to insert transposes
                    # Input: NCHW -> NHWC (perm: 0, 2, 3, 1)
                    # Weight: OIHW -> HWIO (perm: 2, 3, 1, 0)
                    # But for simplicity, we just transpose the inputs and outputs
                    new_attrs = dict(expr.attrs) if expr.attrs else {}
                    new_attrs["layout"] = "NHWC"

                    data_transposed = self._create_transpose(new_args[0], [0, 2, 3, 1])
                    weight_transposed = self._create_transpose(new_args[1], [2, 3, 1, 0])

                    new_call = Call(
                        op=new_op, args=[data_transposed, weight_transposed], attrs=new_attrs
                    )

                    # Output needs to be transformed back if the downstream expects NCHW
                    # This pass should ideally propagate layouts rather than naively injecting both.
                    # A proper layout pass propagates requirements. Here we just inject local transposes.
                    return self._create_transpose(new_call, [0, 3, 1, 2])

        if new_op is not expr.op or any(a is not b for a, b in zip(new_args, expr.args)):
            return Call(op=new_op, args=new_args, attrs=expr.attrs)
        return expr


def transform_layout(expr: Expr, src_layout: str = "NCHW", dst_layout: str = "NHWC") -> Expr:
    """Pass to transform layouts."""
    return LayoutTransform(src_layout, dst_layout).visit(expr)
