from typing import Any, List, Optional, Tuple


class Expr:
    """Base class for TIR Expressions."""

    dtype: str


class Var(Expr):
    def __init__(self, name: str, dtype: str = "int32"):
        self.name = name
        self.dtype = dtype

    def __repr__(self):
        return f"{self.name}: {self.dtype}"


class IntImm(Expr):
    def __init__(self, dtype: str, value: int):
        self.dtype = dtype
        self.value = value


class FloatImm(Expr):
    def __init__(self, dtype: str, value: float):
        self.dtype = dtype
        self.value = value


class StringImm(Expr):
    def __init__(self, value: str):
        self.dtype = "handle"
        self.value = value


class BinaryOp(Expr):
    def __init__(self, a: Expr, b: Expr):
        self.a = a
        self.b = b
        self.dtype = a.dtype


class Add(BinaryOp):
    pass


class Sub(BinaryOp):
    pass


class Mul(BinaryOp):
    pass


class Div(BinaryOp):
    pass


class Mod(BinaryOp):
    pass


class EQ(BinaryOp):
    dtype = "bool"


class NE(BinaryOp):
    dtype = "bool"


class LT(BinaryOp):
    dtype = "bool"


class LE(BinaryOp):
    dtype = "bool"


class GT(BinaryOp):
    dtype = "bool"


class GE(BinaryOp):
    dtype = "bool"


class And(BinaryOp):
    dtype = "bool"


class Or(BinaryOp):
    dtype = "bool"


class Xor(BinaryOp):
    dtype = "bool"


class Call(Expr):
    def __init__(self, dtype: str, op: str, args: list[Expr]):
        self.dtype = dtype
        self.op = op
        self.args = args


class Load(Expr):
    def __init__(self, dtype: str, buffer_var: Var, index: Expr, predicate: Optional[Expr] = None):
        self.dtype = dtype
        self.buffer_var = buffer_var
        self.index = index
        self.predicate = predicate
