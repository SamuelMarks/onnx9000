"""TVM submodule for AST and optimization."""

from typing import Optional

from ..expr import Call, Constant, Expr, Function, If, Let, Op, TupleExpr, TupleGetItem, Var
from ..ty import FuncType, TensorType, TupleType, Type
from ..visitor import ExprVisitor


class TypeChecker(ExprVisitor):
    """Infers shapes and dtypes for all nodes in the AST."""

    def __init__(self):
        """Magic method."""
        """Initialize."""
        """Do the function."""
        # Environment for variable types
        self.env: dict[str, Type] = {}
        # Op registries
        self.op_infer: dict[str, callable] = {}

    def register_op_infer(self, op_name: str, infer_func: callable):
        """Do the function."""
        self.op_infer[op_name] = infer_func

    def visit_var(self, expr: Var) -> Type:
        """Do the function."""
        if expr.type_annotation:
            expr.checked_type = expr.type_annotation
        elif expr.name_hint in self.env:
            expr.checked_type = self.env[expr.name_hint]
        else:
            raise ValueError(f"Unknown type for variable {expr.name_hint}")
        return expr.checked_type

    def visit_constant(self, expr: Constant) -> Type:
        """Do the function."""
        if expr.type_annotation:
            expr.checked_type = expr.type_annotation
        else:
            # Try to infer from data
            if hasattr(expr.data, "shape") and hasattr(expr.data, "dtype"):
                expr.checked_type = TensorType(
                    shape=tuple(expr.data.shape), dtype=str(expr.data.dtype)
                )
            else:
                # Default scalar
                expr.checked_type = TensorType(shape=(), dtype="float32")
        return expr.checked_type

    def visit_op(self, expr: Op) -> Type:
        """Do the function."""
        # Ops themselves don't have a direct type without arguments in our simplified model
        # We handle this in visit_call
        return None

    def visit_call(self, expr: Call) -> Type:
        """Do the function."""
        arg_types = [self.visit(arg) for arg in expr.args]

        if isinstance(expr.op, Op):
            if expr.op.name in self.op_infer:
                ret_type = self.op_infer[expr.op.name](arg_types, expr.attrs or {})
                expr.checked_type = ret_type
                return ret_type
            else:
                # Mock default: return same as first arg
                if arg_types:
                    expr.checked_type = arg_types[0]
                    return arg_types[0]
                else:
                    raise ValueError(f"No type inference for {expr.op.name}")
        elif isinstance(expr.op, Function):
            func_type = self.visit(expr.op)
            if isinstance(func_type, FuncType):
                expr.checked_type = func_type.ret_type
                return func_type.ret_type

        raise ValueError("Invalid call operator")

    def visit_tuple(self, expr: TupleExpr) -> Type:
        """Do the function."""
        field_types = [self.visit(field) for field in expr.fields]
        expr.checked_type = TupleType(fields=field_types)
        return expr.checked_type

    def visit_tuple_getitem(self, expr: TupleGetItem) -> Type:
        """Do the function."""
        tuple_type = self.visit(expr.tuple_value)
        if isinstance(tuple_type, TupleType):
            if 0 <= expr.index < len(tuple_type.fields):
                expr.checked_type = tuple_type.fields[expr.index]
                return expr.checked_type
            raise IndexError(f"tuple index {expr.index} out of bounds")
        raise TypeError("Expected TupleType")

    def visit_let(self, expr: Let) -> Type:
        """Do the function."""
        val_type = self.visit(expr.value)

        # Save old env to restore scope later
        old_type = self.env.get(expr.var.name_hint)
        self.env[expr.var.name_hint] = val_type

        expr.var.checked_type = val_type

        body_type = self.visit(expr.body)
        expr.checked_type = body_type

        if old_type is not None:
            self.env[expr.var.name_hint] = old_type
        else:
            del self.env[expr.var.name_hint]

        return expr.checked_type

    def visit_if(self, expr: If) -> Type:
        """Do the function."""
        self.visit(expr.cond)
        # Should check if cond is bool/int
        true_type = self.visit(expr.true_branch)
        self.visit(expr.false_branch)
        # Assuming true/false branches have same type (simple)
        expr.checked_type = true_type
        return expr.checked_type

    def visit_function(self, expr: Function) -> Type:
        """Do the function."""
        arg_types = []
        old_env = {}

        for param in expr.params:
            if param.type_annotation:
                arg_types.append(param.type_annotation)
                param.checked_type = param.type_annotation
                old_env[param.name_hint] = self.env.get(param.name_hint)
                self.env[param.name_hint] = param.type_annotation
            else:
                raise ValueError(f"Function parameter {param.name_hint} missing type annotation")

        ret_type = self.visit(expr.body)

        if expr.ret_type and expr.ret_type != ret_type:
            # Very naive equality check, in practice need Type matching
            pass

        # Restore environment
        for k, v in old_env.items():
            if v is not None:
                self.env[k] = v
            else:
                del self.env[k]

        func_type = FuncType(arg_types=arg_types, ret_type=ret_type)
        expr.checked_type = func_type
        return func_type


def infer_type(expr: Expr, op_infer: dict[str, callable] = None) -> Expr:
    """Pass to infer shapes and types."""
    checker = TypeChecker()
    if op_infer:
        for op, func in op_infer.items():
            checker.register_op_infer(op, func)
    checker.visit(expr)
    return expr
