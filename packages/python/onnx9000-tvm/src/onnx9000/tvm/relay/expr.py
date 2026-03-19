from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .ty import FuncType, Type


class Expr:
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    """Base class for all Relay expressions."""

    checked_type: Optional[Type] = None
    span: Optional[Any] = None


@dataclass(eq=False)
class Var(Expr):
    """A local variable in Relay."""

    name_hint: str
    type_annotation: Optional[Type] = None


@dataclass(eq=False)
class Constant(Expr):
    """A constant tensor in Relay."""

    data: Any  # Can be a raw bytes/memoryview or numpy-like object
    type_annotation: Optional[Type] = None


@dataclass(eq=False)
class Op(Expr):
    """An operator declaration."""

    name: str


@dataclass(eq=False)
class Call(Expr):
    """A function/operator invocation."""

    op: Union[Op, Expr]
    args: list[Expr]
    attrs: Optional[dict[str, Any]] = None


@dataclass(eq=False)
class TupleExpr(Expr):
    """A tuple of expressions."""

    fields: list[Expr]


@dataclass(eq=False)
class TupleGetItem(Expr):
    """Get an element of a tuple."""

    tuple_value: Expr
    index: int


@dataclass(eq=False)
class Let(Expr):
    """A let binding for local variable scope."""

    var: Var
    value: Expr
    body: Expr


@dataclass(eq=False)
class If(Expr):
    """A conditional expression."""

    cond: Expr
    true_branch: Expr
    false_branch: Expr


@dataclass(eq=False)
class Function(Expr):
    """A global or local function."""

    params: list[Var]
    body: Expr
    ret_type: Optional[Type] = None
    type_params: list[Type] = field(default_factory=list)
