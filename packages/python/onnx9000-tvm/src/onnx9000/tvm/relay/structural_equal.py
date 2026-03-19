from typing import Dict, Tuple

from .expr import Call, Constant, Expr, Function, If, Let, Op, TupleExpr, TupleGetItem, Var


class StructuralEquality:
    """
    Checks if two Relay expressions are structurally equal.
    """

    def __init__(self):
        self.var_map: dict[str, str] = {}

    def equal(self, a: Expr, b: Expr) -> bool:
        if type(a) != type(b):
            return False

        if isinstance(a, Var):
            # Same variable or mapped variable
            return self.var_map.get(a.name_hint, a.name_hint) == b.name_hint

        elif isinstance(a, Constant):
            # Check data equality
            try:
                import numpy as np

                return np.array_equal(a.data, b.data)
            except:
                return a.data == b.data

        elif isinstance(a, Op):
            return a.name == b.name

        elif isinstance(a, Call):
            if not self.equal(a.op, b.op):
                return False
            if len(a.args) != len(b.args):
                return False
            for aa, bb in zip(a.args, b.args):
                if not self.equal(aa, bb):
                    return False
            if a.attrs != b.attrs:
                return False
            return True

        elif isinstance(a, TupleExpr):
            if len(a.fields) != len(b.fields):
                return False
            return all(self.equal(aa, bb) for aa, bb in zip(a.fields, b.fields))

        elif isinstance(a, TupleGetItem):
            return a.index == b.index and self.equal(a.tuple_value, b.tuple_value)

        elif isinstance(a, Let):
            # Map a.var to b.var
            old_val = self.var_map.get(a.var.name_hint)
            self.var_map[a.var.name_hint] = b.var.name_hint

            val_eq = self.equal(a.value, b.value)
            if not val_eq:
                return False

            body_eq = self.equal(a.body, b.body)

            # Restore map
            if old_val is not None:
                self.var_map[a.var.name_hint] = old_val
            else:
                del self.var_map[a.var.name_hint]

            return body_eq

        elif isinstance(a, If):
            return (
                self.equal(a.cond, b.cond)
                and self.equal(a.true_branch, b.true_branch)
                and self.equal(a.false_branch, b.false_branch)
            )

        elif isinstance(a, Function):
            if len(a.params) != len(b.params):
                return False
            old_vals = {}
            for ap, bp in zip(a.params, b.params):
                old_vals[ap.name_hint] = self.var_map.get(ap.name_hint)
                self.var_map[ap.name_hint] = bp.name_hint

            body_eq = self.equal(a.body, b.body)

            for ap in a.params:
                if old_vals[ap.name_hint] is not None:
                    self.var_map[ap.name_hint] = old_vals[ap.name_hint]
                else:
                    del self.var_map[ap.name_hint]

            return body_eq

        return False


def structural_equal(a: Expr, b: Expr) -> bool:
    """Returns True if the two expressions are structurally equal."""
    return StructuralEquality().equal(a, b)
