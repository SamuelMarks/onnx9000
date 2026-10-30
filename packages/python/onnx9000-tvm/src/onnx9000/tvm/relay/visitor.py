"""Relay AST expression visitors and mutators."""

from __future__ import annotations

from .expr import Call, Constant, Expr, Function, If, Let, Op, TupleExpr, TupleGetItem, Var
from .ty import Type

VisitResult = str | Type | None


class ExprVisitor:
    """Base class for all expression visitors."""

    def visit(self, expr: Expr) -> VisitResult:
        """Dispatch visitor to the specific expression node handler."""
        if isinstance(expr, Var):
            return self.visit_var(expr)
        if isinstance(expr, Constant):
            return self.visit_constant(expr)
        if isinstance(expr, Op):
            return self.visit_op(expr)
        if isinstance(expr, Call):
            return self.visit_call(expr)
        if isinstance(expr, TupleExpr):
            return self.visit_tuple(expr)
        if isinstance(expr, TupleGetItem):
            return self.visit_tuple_getitem(expr)
        if isinstance(expr, Let):
            return self.visit_let(expr)
        if isinstance(expr, If):
            return self.visit_if(expr)
        if isinstance(expr, Function):
            return self.visit_function(expr)
        return None

    def visit_var(self, expr: Var) -> VisitResult:
        """Visit a Var expression."""
        return None

    def visit_constant(self, expr: Constant) -> VisitResult:
        """Visit a Constant expression."""
        return None

    def visit_op(self, expr: Op) -> VisitResult:
        """Visit an Op expression."""
        return None

    def visit_call(self, expr: Call) -> VisitResult:
        """Visit a Call expression."""
        self.visit(expr.op)
        for arg in expr.args:
            self.visit(arg)
        return None

    def visit_tuple(self, expr: TupleExpr) -> VisitResult:
        """Visit a TupleExpr expression."""
        for field in expr.fields:
            self.visit(field)
        return None

    def visit_tuple_getitem(self, expr: TupleGetItem) -> VisitResult:
        """Visit a TupleGetItem expression."""
        self.visit(expr.tuple_value)
        return None

    def visit_let(self, expr: Let) -> VisitResult:
        """Visit a Let expression."""
        self.visit(expr.var)
        self.visit(expr.value)
        self.visit(expr.body)
        return None

    def visit_if(self, expr: If) -> VisitResult:
        """Visit an If expression."""
        self.visit(expr.cond)
        self.visit(expr.true_branch)
        self.visit(expr.false_branch)
        return None

    def visit_function(self, expr: Function) -> VisitResult:
        """Visit a Function expression."""
        for param in expr.params:
            self.visit(param)
        self.visit(expr.body)
        return None


class ExprMutator:
    """Base class for all expression mutators."""

    def visit(self, expr: Expr) -> Expr:
        """Dispatch mutator to the specific expression node handler."""
        if isinstance(expr, Var):
            return self.visit_var(expr)
        if isinstance(expr, Constant):
            return self.visit_constant(expr)
        if isinstance(expr, Op):
            return self.visit_op(expr)
        if isinstance(expr, Call):
            return self.visit_call(expr)
        if isinstance(expr, TupleExpr):
            return self.visit_tuple(expr)
        if isinstance(expr, TupleGetItem):
            return self.visit_tuple_getitem(expr)
        if isinstance(expr, Let):
            return self.visit_let(expr)
        if isinstance(expr, If):
            return self.visit_if(expr)
        if isinstance(expr, Function):
            return self.visit_function(expr)
        return expr

    def visit_var(self, expr: Var) -> Expr:
        """Mutate a Var expression."""
        return expr

    def visit_constant(self, expr: Constant) -> Expr:
        """Mutate a Constant expression."""
        return expr

    def visit_op(self, expr: Op) -> Expr:
        """Mutate an Op expression."""
        return expr

    def visit_call(self, expr: Call) -> Expr:
        """Mutate a Call expression."""
        new_op = self.visit(expr.op)
        new_args = [self.visit(arg) for arg in expr.args]
        if new_op is not expr.op or any(
            a is not b for a, b in zip(new_args, expr.args, strict=False)
        ):
            return Call(op=new_op, args=new_args, attrs=expr.attrs)
        return expr

    def visit_tuple(self, expr: TupleExpr) -> Expr:
        """Mutate a TupleExpr expression."""
        new_fields = [self.visit(field) for field in expr.fields]
        if any(a is not b for a, b in zip(new_fields, expr.fields, strict=False)):
            return TupleExpr(fields=new_fields)
        return expr

    def visit_tuple_getitem(self, expr: TupleGetItem) -> Expr:
        """Mutate a TupleGetItem expression."""
        new_tuple_value = self.visit(expr.tuple_value)
        if new_tuple_value is not expr.tuple_value:
            return TupleGetItem(tuple_value=new_tuple_value, index=expr.index)
        return expr

    def visit_let(self, expr: Let) -> Expr:
        """Mutate a Let expression."""
        new_var = self.visit(expr.var)
        new_value = self.visit(expr.value)
        new_body = self.visit(expr.body)
        if new_var is not expr.var or new_value is not expr.value or new_body is not expr.body:
            var_node = new_var if isinstance(new_var, Var) else expr.var
            return Let(var=var_node, value=new_value, body=new_body)
        return expr

    def visit_if(self, expr: If) -> Expr:
        """Mutate an If expression."""
        new_cond = self.visit(expr.cond)
        new_true_branch = self.visit(expr.true_branch)
        new_false_branch = self.visit(expr.false_branch)
        if (
            new_cond is not expr.cond
            or new_true_branch is not expr.true_branch
            or new_false_branch is not expr.false_branch
        ):
            return If(cond=new_cond, true_branch=new_true_branch, false_branch=new_false_branch)
        return expr

    def visit_function(self, expr: Function) -> Expr:
        """Mutate a Function expression."""
        new_params = [self.visit(param) for param in expr.params]
        new_body = self.visit(expr.body)
        if (
            any(a is not b for a, b in zip(new_params, expr.params, strict=False))
            or new_body is not expr.body
        ):
            typed_params = [
                p if isinstance(p, Var) else expr.params[i] for i, p in enumerate(new_params)
            ]
            return Function(
                params=typed_params,
                body=new_body,
                ret_type=expr.ret_type,
                type_params=expr.type_params,
            )
        return expr
