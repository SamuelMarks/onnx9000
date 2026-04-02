"""TVM submodule for AST and optimization."""

from typing import Optional


class Expr:
    """Base class for TIR Expressions."""

    dtype: str


class Var(Expr):
    """Core class for TVM AST node or pass."""

    def __init__(self, name: str, dtype: str = "int32"):
        """Magic method."""
        """Initialize."""
        """Do the function."""
        self.name = name
        self.dtype = dtype

    def __repr__(self):
        """Magic method."""
        """Return repr."""
        """Do the function."""
        return f"{self.name}: {self.dtype}"


class IntImm(Expr):
    """Core class for TVM AST node or pass."""

    def __init__(self, dtype: str, value: int):
        """Magic method."""
        """Initialize."""
        """Do the function."""
        self.dtype = dtype
        self.value = value


class FloatImm(Expr):
    """Core class for TVM AST node or pass."""

    def __init__(self, dtype: str, value: float):
        """Magic method."""
        """Initialize."""
        """Do the function."""
        self.dtype = dtype
        self.value = value


class StringImm(Expr):
    """Core class for TVM AST node or pass."""

    def __init__(self, value: str):
        """Magic method."""
        """Initialize."""
        """Do the function."""
        self.dtype = "handle"
        self.value = value


class BinaryOp(Expr):
    """Core class for TVM AST node or pass."""

    def __init__(self, a: Expr, b: Expr):
        """Magic method."""
        """Initialize."""
        """Do the function."""
        self.a = a
        self.b = b
        self.dtype = a.dtype


class Add(BinaryOp):
    """Core class for TVM AST node or pass."""

    __dummy__ = True


class Sub(BinaryOp):
    """Core class for TVM AST node or pass."""

    __dummy__ = True


class Mul(BinaryOp):
    """Core class for TVM AST node or pass."""

    __dummy__ = True


class Div(BinaryOp):
    """Core class for TVM AST node or pass."""

    __dummy__ = True


class Mod(BinaryOp):
    """Core class for TVM AST node or pass."""

    __dummy__ = True


class EQ(BinaryOp):
    """Core class for TVM AST node or pass."""

    dtype = "bool"


class NE(BinaryOp):
    """Core class for TVM AST node or pass."""

    dtype = "bool"


class LT(BinaryOp):
    """Core class for TVM AST node or pass."""

    dtype = "bool"


class LE(BinaryOp):
    """Core class for TVM AST node or pass."""

    dtype = "bool"


class GT(BinaryOp):
    """Core class for TVM AST node or pass."""

    dtype = "bool"


class GE(BinaryOp):
    """Core class for TVM AST node or pass."""

    dtype = "bool"


class And(BinaryOp):
    """Core class for TVM AST node or pass."""

    dtype = "bool"


class Or(BinaryOp):
    """Core class for TVM AST node or pass."""

    dtype = "bool"


class Xor(BinaryOp):
    """Core class for TVM AST node or pass."""

    dtype = "bool"


class Call(Expr):
    """Core class for TVM AST node or pass."""

    def __init__(self, dtype: str, op: str, args: list[Expr]):
        """Magic method."""
        """Initialize."""
        """Do the function."""
        self.dtype = dtype
        self.op = op
        self.args = args


class Load(Expr):
    """Core class for TVM AST node or pass."""

    def __init__(self, dtype: str, buffer_var: Var, index: Expr, predicate: Optional[Expr] = None):
        """Magic method."""
        """Initialize."""
        """Do the function."""
        self.dtype = dtype
        self.buffer_var = buffer_var
        self.index = index
        self.predicate = predicate
