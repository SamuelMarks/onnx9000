"""TVM submodule for AST and optimization."""

from typing import Any

from .expr import *
from .stmt import *


class StmtVisitor:
    """Core class for TVM AST node or pass."""

    def visit(self, stmt: Stmt) -> Any:
        """Do the function."""
        if isinstance(stmt, LetStmt):
            return self.visit_LetStmt(stmt)
        elif isinstance(stmt, AssertStmt):
            return self.visit_AssertStmt(stmt)
        elif isinstance(stmt, For):
            return self.visit_For(stmt)
        elif isinstance(stmt, While):
            return self.visit_While(stmt)
        elif isinstance(stmt, Store):
            return self.visit_Store(stmt)
        elif isinstance(stmt, Allocate):
            return self.visit_Allocate(stmt)
        elif isinstance(stmt, IfThenElse):
            return self.visit_IfThenElse(stmt)
        elif isinstance(stmt, Evaluate):
            return self.visit_Evaluate(stmt)
        elif isinstance(stmt, SeqStmt):
            return self.visit_SeqStmt(stmt)
        else:
            raise NotImplementedError(f"Visitor for {type(stmt)} not implemented")

    def visit_LetStmt(self, stmt: LetStmt):
        """Do the function."""
        self.visit(stmt.body)

    def visit_AssertStmt(self, stmt: AssertStmt):
        """Do the function."""
        self.visit(stmt.body)

    def visit_For(self, stmt: For):
        """Do the function."""
        self.visit(stmt.body)

    def visit_While(self, stmt: While):
        """Do the function."""
        self.visit(stmt.body)

    def visit_Store(self, stmt: Store):
        """Do the function."""
        pass

    def visit_Allocate(self, stmt: Allocate):
        """Do the function."""
        self.visit(stmt.body)

    def visit_IfThenElse(self, stmt: IfThenElse):
        """Do the function."""
        self.visit(stmt.then_case)
        if stmt.else_case:
            self.visit(stmt.else_case)

    def visit_Evaluate(self, stmt: Evaluate):
        """Do the function."""
        pass

    def visit_SeqStmt(self, stmt: SeqStmt):
        """Do the function."""
        for s in stmt.seq:
            self.visit(s)


class StmtMutator:
    """Core class for TVM AST node or pass."""

    def visit(self, stmt: Stmt) -> Stmt:
        """Do the function."""
        if isinstance(stmt, LetStmt):
            return self.visit_LetStmt(stmt)
        elif isinstance(stmt, AssertStmt):
            return self.visit_AssertStmt(stmt)
        elif isinstance(stmt, For):
            return self.visit_For(stmt)
        elif isinstance(stmt, While):
            return self.visit_While(stmt)
        elif isinstance(stmt, Store):
            return self.visit_Store(stmt)
        elif isinstance(stmt, Allocate):
            return self.visit_Allocate(stmt)
        elif isinstance(stmt, IfThenElse):
            return self.visit_IfThenElse(stmt)
        elif isinstance(stmt, Evaluate):
            return self.visit_Evaluate(stmt)
        elif isinstance(stmt, SeqStmt):
            return self.visit_SeqStmt(stmt)
        else:
            raise NotImplementedError(f"Mutator for {type(stmt)} not implemented")

    def visit_LetStmt(self, stmt: LetStmt):
        """Do the function."""
        body = self.visit(stmt.body)
        if body is stmt.body:
            return stmt
        return LetStmt(stmt.var, stmt.value, body)

    def visit_AssertStmt(self, stmt: AssertStmt):
        """Do the function."""
        body = self.visit(stmt.body)
        if body is stmt.body:
            return stmt
        return AssertStmt(stmt.condition, stmt.message, body)

    def visit_For(self, stmt: For):
        """Do the function."""
        body = self.visit(stmt.body)
        if body is stmt.body:
            return stmt
        return For(stmt.loop_var, stmt.min_val, stmt.extent, stmt.kind, body)

    def visit_While(self, stmt: While):
        """Do the function."""
        body = self.visit(stmt.body)
        if body is stmt.body:
            return stmt
        return While(stmt.condition, body)

    def visit_Store(self, stmt: Store):
        """Do the function."""
        return stmt

    def visit_Allocate(self, stmt: Allocate):
        """Do the function."""
        body = self.visit(stmt.body)
        if body is stmt.body:
            return stmt
        return Allocate(stmt.buffer_var, stmt.dtype, stmt.extents, stmt.condition, body)

    def visit_IfThenElse(self, stmt: IfThenElse):
        """Do the function."""
        then_case = self.visit(stmt.then_case)
        else_case = self.visit(stmt.else_case) if stmt.else_case else None
        if then_case is stmt.then_case and else_case is stmt.else_case:
            return stmt
        return IfThenElse(stmt.condition, then_case, else_case)

    def visit_Evaluate(self, stmt: Evaluate):
        """Do the function."""
        return stmt

    def visit_SeqStmt(self, stmt: SeqStmt):
        """Do the function."""
        seq = [self.visit(s) for s in stmt.seq]
        if all(a is b for a, b in zip(seq, stmt.seq)):
            return stmt
        return SeqStmt(seq)
