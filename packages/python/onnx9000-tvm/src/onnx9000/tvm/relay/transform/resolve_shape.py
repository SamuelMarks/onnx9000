"""Relay dynamic shape resolution pass."""

from __future__ import annotations

from ..expr import Expr, Function, Var
from ..ty import FuncType, TensorType, TupleType, Type
from ..visitor import ExprMutator


class ShapeResolver(ExprMutator):
    """Resolves dynamic shape components (strings) to static shapes based on bounds."""

    def __init__(self, bounds: dict[str, int]) -> None:
        """Initialize the shape resolver with symbolic dimension bounds."""
        self.bounds = bounds

    def _resolve_type(self, ty: Type) -> Type:
        """Resolve dynamic dimensions inside a Relay Type."""
        if isinstance(ty, TensorType):
            new_shape: list[int | str] = []
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
        if isinstance(ty, TupleType):
            new_fields = [self._resolve_type(f) for f in ty.fields]
            if any(n is not o for n, o in zip(new_fields, ty.fields, strict=False)):
                return TupleType(fields=new_fields)
            return ty
        if isinstance(ty, FuncType):
            new_args = [self._resolve_type(a) for a in ty.arg_types]
            new_ret = self._resolve_type(ty.ret_type)
            if (
                any(n is not o for n, o in zip(new_args, ty.arg_types, strict=False))
                or new_ret is not ty.ret_type
            ):
                return FuncType(arg_types=new_args, ret_type=new_ret)
            return ty
        return ty

    def visit(self, expr: Expr) -> Expr:
        """Visit and resolve dynamic types on an expression."""
        new_expr = super().visit(expr)
        if new_expr.checked_type is not None:
            new_expr.checked_type = self._resolve_type(new_expr.checked_type)
        return new_expr

    def visit_var(self, expr: Var) -> Expr:
        """Resolve dynamic dimensions on a Var type annotation."""
        if expr.type_annotation:
            new_ty = self._resolve_type(expr.type_annotation)
            if new_ty is not expr.type_annotation:
                return Var(name_hint=expr.name_hint, type_annotation=new_ty)
        return expr

    def visit_function(self, expr: Function) -> Expr:
        """Resolve dynamic dimensions on Function parameters and return type."""
        new_params = [self.visit(p) for p in expr.params]
        new_body = self.visit(expr.body)
        new_ret_type = self._resolve_type(expr.ret_type) if expr.ret_type else None

        if (
            any(n is not o for n, o in zip(new_params, expr.params, strict=False))
            or new_body is not expr.body
            or new_ret_type is not expr.ret_type
        ):
            typed_params = [
                p if isinstance(p, Var) else expr.params[i] for i, p in enumerate(new_params)
            ]
            return Function(
                params=typed_params,
                body=new_body,
                ret_type=new_ret_type,
                type_params=expr.type_params,
            )
        return expr


def resolve_dynamic_shape(expr: Expr, bounds: dict[str, int]) -> Expr:
    """Pass to resolve dynamic to static shape."""
    return ShapeResolver(bounds).visit(expr)
