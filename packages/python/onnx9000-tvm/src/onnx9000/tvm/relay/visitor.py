"""TVM submodule for AST and optimization."""

from typing import Any

from .expr import Call, Constant, Expr, Function, If, Let, Op, TupleExpr, TupleGetItem, Var


class ExprVisitor:
    """Base class for all expression visitors."""

    def visit(self, expr: Expr) -> Any:
        """Do the function."""
        if isinstance(expr, Var):
            return self.visit_var(expr)
        elif isinstance(expr, Constant):
            return self.visit_constant(expr)
        elif isinstance(expr, Op):
            return self.visit_op(expr)
        elif isinstance(expr, Call):
            return self.visit_call(expr)
        elif isinstance(expr, TupleExpr):
            return self.visit_tuple(expr)
        elif isinstance(expr, TupleGetItem):
            return self.visit_tuple_getitem(expr)
        elif isinstance(expr, Let):
            return self.visit_let(expr)
        elif isinstance(expr, If):
            return self.visit_if(expr)
        elif isinstance(expr, Function):
            return self.visit_function(expr)
        else:
            return None

    def visit_var(self, expr: Var) -> Any:
        """Do the function."""
        return None

    def visit_constant(self, expr: Constant) -> Any:
        """Do the function."""
        return None

    def visit_op(self, expr: Op) -> Any:
        """Do the function."""
        return None

    def visit_call(self, expr: Call) -> Any:
        """Do the function."""
        self.visit(expr.op)
        for arg in expr.args:
            self.visit(arg)

    def visit_tuple(self, expr: TupleExpr) -> Any:
        """Do the function."""
        for field in expr.fields:
            self.visit(field)

    def visit_tuple_getitem(self, expr: TupleGetItem) -> Any:
        """Do the function."""
        self.visit(expr.tuple_value)

    def visit_let(self, expr: Let) -> Any:
        """Do the function."""
        self.visit(expr.var)
        self.visit(expr.value)
        self.visit(expr.body)

    def visit_if(self, expr: If) -> Any:
        """Do the function."""
        self.visit(expr.cond)
        self.visit(expr.true_branch)
        self.visit(expr.false_branch)

    def visit_function(self, expr: Function) -> Any:
        """Do the function."""
        for param in expr.params:
            self.visit(param)
        self.visit(expr.body)


class ExprMutator:
    """Base class for all expression mutators."""

    def visit(self, expr: Expr) -> Expr:
        """Do the function."""
        if isinstance(expr, Var):
            return self.visit_var(expr)
        elif isinstance(expr, Constant):
            return self.visit_constant(expr)
        elif isinstance(expr, Op):
            return self.visit_op(expr)
        elif isinstance(expr, Call):
            return self.visit_call(expr)
        elif isinstance(expr, TupleExpr):
            return self.visit_tuple(expr)
        elif isinstance(expr, TupleGetItem):
            return self.visit_tuple_getitem(expr)
        elif isinstance(expr, Let):
            return self.visit_let(expr)
        elif isinstance(expr, If):
            return self.visit_if(expr)
        elif isinstance(expr, Function):
            return self.visit_function(expr)
        else:
            return None

    def visit_var(self, expr: Var) -> Expr:
        """Do the function."""
        return expr

    def visit_constant(self, expr: Constant) -> Expr:
        """Do the function."""
        return expr

    def visit_op(self, expr: Op) -> Expr:
        """Do the function."""
        return expr

    def visit_call(self, expr: Call) -> Expr:
        """Do the function."""
        new_op = self.visit(expr.op)
        new_args = [self.visit(arg) for arg in expr.args]
        if new_op is not expr.op or any(a is not b for a, b in zip(new_args, expr.args)):
            return Call(op=new_op, args=new_args, attrs=expr.attrs)
        return expr

    def visit_tuple(self, expr: TupleExpr) -> Expr:
        """Do the function."""
        new_fields = [self.visit(field) for field in expr.fields]
        if any(a is not b for a, b in zip(new_fields, expr.fields)):
            return TupleExpr(fields=new_fields)
        return expr

    def visit_tuple_getitem(self, expr: TupleGetItem) -> Expr:
        """Do the function."""
        new_tuple_value = self.visit(expr.tuple_value)
        if new_tuple_value is not expr.tuple_value:
            return TupleGetItem(tuple_value=new_tuple_value, index=expr.index)
        return expr

    def visit_let(self, expr: Let) -> Expr:
        """Do the function."""
        new_var = self.visit(expr.var)
        new_value = self.visit(expr.value)
        new_body = self.visit(expr.body)
        if new_var is not expr.var or new_value is not expr.value or new_body is not expr.body:
            return Let(var=new_var, value=new_value, body=new_body)
        return expr

    def visit_if(self, expr: If) -> Expr:
        """Do the function."""
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
        """Do the function."""
        new_params = [self.visit(param) for param in expr.params]
        new_body = self.visit(expr.body)
        if any(a is not b for a, b in zip(new_params, expr.params)) or new_body is not expr.body:
            return Function(
                params=new_params,
                body=new_body,
                ret_type=expr.ret_type,
                type_params=expr.type_params,
            )
        return expr
