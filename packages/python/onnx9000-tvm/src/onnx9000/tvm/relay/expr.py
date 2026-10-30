"""Relay expression AST nodes for TVM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .span import Span
from .ty import Type

_RUNTIME_TYPES = (Span, Type)


@runtime_checkable
class ShapedDataProtocol(Protocol):
    """Protocol for tensor-like constant data with shape and dtype."""

    shape: tuple[int, ...]
    dtype: str


ConstantValue = (
    int
    | float
    | bool
    | str
    | bytes
    | memoryview
    | list[int | float | bool]
    | tuple[int | float | bool, ...]
    | ShapedDataProtocol
    | None
)

AttrValue = int | float | str | bool | list[int] | list[float] | list[str] | tuple[int, ...] | None


class Expr:
    """Base class for all Relay expressions."""

    checked_type: Type | None = None
    span: Span | None = None

    def __hash__(self) -> int:
        """Compute identity-based hash of the expression."""
        return id(self)

    def __eq__(self, other: Expr) -> bool:  # type: ignore[override]
        """Compare identity equality of two expressions."""
        return self is other


@dataclass(eq=False)
class Var(Expr):
    """A local variable in Relay."""

    name_hint: str
    type_annotation: Type | None = None


@dataclass(eq=False)
class Constant(Expr):
    """A constant tensor in Relay."""

    data: ConstantValue
    type_annotation: Type | None = None


@dataclass(eq=False)
class Op(Expr):
    """An operator declaration."""

    name: str


@dataclass(eq=False)
class Call(Expr):
    """A function/operator invocation."""

    op: Op | Expr
    args: list[Expr]
    attrs: dict[str, AttrValue] | None = None


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
    ret_type: Type | None = None
    type_params: list[Type] = field(default_factory=list)
