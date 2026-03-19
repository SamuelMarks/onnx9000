"""TVM submodule for AST and optimization."""

from typing import Any, Union

from ..expr import Call, Constant, Expr, Function, If, Let, Op, TupleExpr, TupleGetItem, Var
from ..ty import FuncType, TensorType, TupleType, Type
from ..visitor import ExprMutator


class ShapeResolver(ExprMutator):
    """Resolves dynamic shape components (strings) to static shapes based on bounds."""

    def __init__(self, bounds: dict[str, int]):
        """Magic method."""
        """Initialize."""
        """Do the function."""
        self.bounds = bounds

    def _resolve_type(self, ty: Type) -> Type:
        """Do the function."""
        if isinstance(ty, TensorType):
            new_shape = []
            changed = False
            for dim in ty.shape:
                if isinstance(dim, str) and dim in self.bounds:
                    new_shape.append(self.bounds[dim])
                    changed = True
                else:
                    new_shape.append(dim)
            if changed:
                return TensorType(shape=tuple(new_shape), dtype=ty.dtype)
            return ty
        elif isinstance(ty, TupleType):
            new_fields = [self._resolve_type(f) for f in ty.fields]
            if any(n is not o for n, o in zip(new_fields, ty.fields)):
                return TupleType(fields=new_fields)
            return ty
        elif isinstance(ty, FuncType):
            new_args = [self._resolve_type(a) for a in ty.arg_types]
            new_ret = self._resolve_type(ty.ret_type)
            if (
                any(n is not o for n, o in zip(new_args, ty.arg_types))
                or new_ret is not ty.ret_type
            ):
                return FuncType(arg_types=new_args, ret_type=new_ret)
            return ty
        return ty

    def visit(self, expr: Expr) -> Expr:
        """Do the function."""
        new_expr = super().visit(expr)

        # Also mutate checked_type
        if getattr(new_expr, "checked_type", None) is not None:
            new_expr.checked_type = self._resolve_type(new_expr.checked_type)

        return new_expr

    def visit_var(self, expr: Var) -> Expr:
        """Do the function."""
        if expr.type_annotation:
            new_ty = self._resolve_type(expr.type_annotation)
            if new_ty is not expr.type_annotation:
                return Var(name_hint=expr.name_hint, type_annotation=new_ty)
        return expr

    def visit_function(self, expr: Function) -> Expr:
        """Do the function."""
        new_params = [self.visit(p) for p in expr.params]
        new_body = self.visit(expr.body)
        new_ret_type = self._resolve_type(expr.ret_type) if expr.ret_type else None

        if (
            any(n is not o for n, o in zip(new_params, expr.params))
            or new_body is not expr.body
            or new_ret_type is not expr.ret_type
        ):
            return Function(
                params=new_params,
                body=new_body,
                ret_type=new_ret_type,
                type_params=expr.type_params,
            )
        return expr


def resolve_dynamic_shape(expr: Expr, bounds: dict[str, int]) -> Expr:
    """Pass to resolve dynamic to static shape."""
    return ShapeResolver(bounds).visit(expr)
