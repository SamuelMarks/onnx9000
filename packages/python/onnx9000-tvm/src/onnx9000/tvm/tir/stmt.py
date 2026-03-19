"""TVM submodule for AST and optimization."""

from typing import Any, Optional

from .expr import Expr, Var


class Stmt:
    """Base class for TIR statements."""

    pass


class LetStmt(Stmt):
    """Evaluates or manipulates TVM AST nodes."""

    def __init__(self, var: Var, value: Expr, body: Stmt):
        """Evaluates or manipulates TVM AST nodes."""
        self.var = var
        self.value = value
        self.body = body


class AssertStmt(Stmt):
    """Evaluates or manipulates TVM AST nodes."""

    def __init__(self, condition: Expr, message: Expr, body: Stmt):
        """Evaluates or manipulates TVM AST nodes."""
        self.condition = condition
        self.message = message
        self.body = body


class For(Stmt):
    """Evaluates or manipulates TVM AST nodes."""

    def __init__(self, loop_var: Var, min_val: Expr, extent: Expr, kind: str, body: Stmt):
        """Evaluates or manipulates TVM AST nodes."""
        self.loop_var = loop_var
        self.min_val = min_val
        self.extent = extent
        self.kind = kind  # "serial", "parallel", "vectorized", "unrolled"
        self.body = body


class While(Stmt):
    """Evaluates or manipulates TVM AST nodes."""

    def __init__(self, condition: Expr, body: Stmt):
        """Evaluates or manipulates TVM AST nodes."""
        self.condition = condition
        self.body = body


class Store(Stmt):
    """Evaluates or manipulates TVM AST nodes."""

    def __init__(self, buffer_var: Var, value: Expr, index: Expr, predicate: Optional[Expr] = None):
        """Evaluates or manipulates TVM AST nodes."""
        self.buffer_var = buffer_var
        self.value = value
        self.index = index
        self.predicate = predicate


class Allocate(Stmt):
    """Evaluates or manipulates TVM AST nodes."""

    def __init__(
        self, buffer_var: Var, dtype: str, extents: list[Expr], condition: Expr, body: Stmt
    ):
        """Evaluates or manipulates TVM AST nodes."""
        self.buffer_var = buffer_var
        self.dtype = dtype
        self.extents = extents
        self.condition = condition
        self.body = body


class IfThenElse(Stmt):
    """Evaluates or manipulates TVM AST nodes."""

    def __init__(self, condition: Expr, then_case: Stmt, else_case: Optional[Stmt] = None):
        """Evaluates or manipulates TVM AST nodes."""
        self.condition = condition
        self.then_case = then_case
        self.else_case = else_case


class Evaluate(Stmt):
    """Evaluates or manipulates TVM AST nodes."""

    def __init__(self, value: Expr):
        """Evaluates or manipulates TVM AST nodes."""
        self.value = value


class SeqStmt(Stmt):
    """Evaluates or manipulates TVM AST nodes."""

    def __init__(self, seq: list[Stmt]):
        """Evaluates or manipulates TVM AST nodes."""
        self.seq = seq


class Buffer:
    """Evaluates or manipulates TVM AST nodes."""

    def __init__(
        self,
        data: Var,
        dtype: str,
        shape: list[Expr],
        strides: list[Expr],
        elem_offset: Expr,
        name: str = "buffer",
    ):
        """Evaluates or manipulates TVM AST nodes."""
        self.data = data
        self.dtype = dtype
        self.shape = shape
        self.strides = strides
        self.elem_offset = elem_offset
        self.name = name
