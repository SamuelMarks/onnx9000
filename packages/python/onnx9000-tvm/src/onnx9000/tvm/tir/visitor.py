"""TIR statement visitors and mutators."""

from __future__ import annotations

from .stmt import (
    Allocate,
    AssertStmt,
    Evaluate,
    For,
    IfThenElse,
    LetStmt,
    SeqStmt,
    Stmt,
    Store,
    While,
)


class StmtVisitor:
    """Base visitor class for TIR statements."""

    def visit(self, stmt: Stmt) -> None:
        """Dispatch visitor to the specific statement handler."""
        if isinstance(stmt, LetStmt):
            self.visit_LetStmt(stmt)
        elif isinstance(stmt, AssertStmt):
            self.visit_AssertStmt(stmt)
        elif isinstance(stmt, For):
            self.visit_For(stmt)
        elif isinstance(stmt, While):
            self.visit_While(stmt)
        elif isinstance(stmt, Store):
            self.visit_Store(stmt)
        elif isinstance(stmt, Allocate):
            self.visit_Allocate(stmt)
        elif isinstance(stmt, IfThenElse):
            self.visit_IfThenElse(stmt)
        elif isinstance(stmt, Evaluate):
            self.visit_Evaluate(stmt)
        elif isinstance(stmt, SeqStmt):
            self.visit_SeqStmt(stmt)

    def visit_LetStmt(self, stmt: LetStmt) -> None:
        """Visit a LetStmt node."""
        self.visit(stmt.body)

    def visit_AssertStmt(self, stmt: AssertStmt) -> None:
        """Visit an AssertStmt node."""
        self.visit(stmt.body)

    def visit_For(self, stmt: For) -> None:
        """Visit a For loop node."""
        self.visit(stmt.body)

    def visit_While(self, stmt: While) -> None:
        """Visit a While loop node."""
        self.visit(stmt.body)

    def visit_Store(self, stmt: Store) -> None:
        """Visit a Store node."""
        return None

    def visit_Allocate(self, stmt: Allocate) -> None:
        """Visit an Allocate node."""
        self.visit(stmt.body)

    def visit_IfThenElse(self, stmt: IfThenElse) -> None:
        """Visit an IfThenElse node."""
        self.visit(stmt.then_case)
        if stmt.else_case:
            self.visit(stmt.else_case)

    def visit_Evaluate(self, stmt: Evaluate) -> None:
        """Visit an Evaluate node."""
        return None

    def visit_SeqStmt(self, stmt: SeqStmt) -> None:
        """Visit a SeqStmt node."""
        for s in stmt.seq:
            self.visit(s)


class StmtMutator:
    """Base mutator class for TIR statements."""

    def visit(self, stmt: Stmt) -> Stmt:
        """Dispatch mutator to the specific statement handler."""
        if isinstance(stmt, LetStmt):
            return self.visit_LetStmt(stmt)
        if isinstance(stmt, AssertStmt):
            return self.visit_AssertStmt(stmt)
        if isinstance(stmt, For):
            return self.visit_For(stmt)
        if isinstance(stmt, While):
            return self.visit_While(stmt)
        if isinstance(stmt, Store):
            return self.visit_Store(stmt)
        if isinstance(stmt, Allocate):
            return self.visit_Allocate(stmt)
        if isinstance(stmt, IfThenElse):
            return self.visit_IfThenElse(stmt)
        if isinstance(stmt, Evaluate):
            return self.visit_Evaluate(stmt)
        if isinstance(stmt, SeqStmt):
            return self.visit_SeqStmt(stmt)
        return stmt

    def visit_LetStmt(self, stmt: LetStmt) -> Stmt:
        """Mutate a LetStmt node."""
        body = self.visit(stmt.body)
        if body is stmt.body:
            return stmt
        return LetStmt(stmt.var, stmt.value, body)

    def visit_AssertStmt(self, stmt: AssertStmt) -> Stmt:
        """Mutate an AssertStmt node."""
        body = self.visit(stmt.body)
        if body is stmt.body:
            return stmt
        return AssertStmt(stmt.condition, stmt.message, body)

    def visit_For(self, stmt: For) -> Stmt:
        """Mutate a For loop node."""
        body = self.visit(stmt.body)
        if body is stmt.body:
            return stmt
        return For(stmt.loop_var, stmt.min_val, stmt.extent, stmt.kind, body)

    def visit_While(self, stmt: While) -> Stmt:
        """Mutate a While loop node."""
        body = self.visit(stmt.body)
        if body is stmt.body:
            return stmt
        return While(stmt.condition, body)

    def visit_Store(self, stmt: Store) -> Stmt:
        """Mutate a Store node."""
        return stmt

    def visit_Allocate(self, stmt: Allocate) -> Stmt:
        """Mutate an Allocate node."""
        body = self.visit(stmt.body)
        if body is stmt.body:
            return stmt
        return Allocate(stmt.buffer_var, stmt.dtype, stmt.extents, stmt.condition, body)

    def visit_IfThenElse(self, stmt: IfThenElse) -> Stmt:
        """Mutate an IfThenElse node."""
        then_case = self.visit(stmt.then_case)
        else_case = self.visit(stmt.else_case) if stmt.else_case else None
        if then_case is stmt.then_case and else_case is stmt.else_case:
            return stmt
        return IfThenElse(stmt.condition, then_case, else_case)

    def visit_Evaluate(self, stmt: Evaluate) -> Stmt:
        """Mutate an Evaluate node."""
        return stmt

    def visit_SeqStmt(self, stmt: SeqStmt) -> Stmt:
        """Mutate a SeqStmt node."""
        seq = [self.visit(s) for s in stmt.seq]
        if all(a is b for a, b in zip(seq, stmt.seq, strict=False)):
            return stmt
        return SeqStmt(seq)
