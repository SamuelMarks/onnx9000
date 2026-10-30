"""TVM Tensor Expression (TE) AST and compute primitives."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from ..relay.expr import AttrValue

_RUNTIME_TYPES = (Callable, AttrValue)


@runtime_checkable
class NamedOpProtocol(Protocol):
    """Protocol for TE operations with a name attribute."""

    name: str


@runtime_checkable
class AxisOpProtocol(Protocol):
    """Protocol for TE operations exposing iteration axes."""

    axis: list[IterVar]


class ExprOp:
    """Base class for math ops on TE variables."""

    def __add__(self, other: ExprOp | int | float) -> Add:
        """Add two TE expressions."""
        return Add(self, other)

    def __radd__(self, other: ExprOp | int | float) -> Add:
        """Right-add two TE expressions."""
        return Add(other, self)

    def __sub__(self, other: ExprOp | int | float) -> Sub:
        """Subtract two TE expressions."""
        return Sub(self, other)

    def __rsub__(self, other: ExprOp | int | float) -> Sub:
        """Right-subtract two TE expressions."""
        return Sub(other, self)

    def __mul__(self, other: ExprOp | int | float) -> Mul:
        """Multiply two TE expressions."""
        return Mul(self, other)

    def __rmul__(self, other: ExprOp | int | float) -> Mul:
        """Right-multiply two TE expressions."""
        return Mul(other, self)

    def __truediv__(self, other: ExprOp | int | float) -> Div:
        """Divide two TE expressions."""
        return Div(self, other)

    def __rtruediv__(self, other: ExprOp | int | float) -> Div:
        """Right-divide two TE expressions."""
        return Div(other, self)


class IterVar(ExprOp):
    """An iteration variable along a tensor dimension."""

    def __init__(self, name: str) -> None:
        """Initialize an iteration variable."""
        self.name = name
        self.dtype = "int32"

    def __repr__(self) -> str:
        """Return string representation of IterVar."""
        return f"IterVar({self.name})"


class ReduceAxis(IterVar):
    """An iteration variable representing a reduction domain."""

    def __init__(self, name: str, dom: tuple[int, int]) -> None:
        """Initialize a reduction axis with domain bounds."""
        super().__init__(name)
        self.dom = dom

    def __repr__(self) -> str:
        """Return string representation of ReduceAxis."""
        return f"ReduceAxis({self.name}, dom={self.dom})"


class Var(ExprOp):
    """A scalar variable in TE compute expressions."""

    def __init__(self, name: str, dtype: str = "int32") -> None:
        """Initialize a TE variable."""
        self.name = name
        self.dtype = dtype

    def __repr__(self) -> str:
        """Return variable name."""
        return self.name


class Const(ExprOp):
    """A scalar constant in TE compute expressions."""

    def __init__(self, val: int | float | str | bool) -> None:
        """Initialize a TE constant."""
        self.val = val

    def __repr__(self) -> str:
        """Return string representation of constant value."""
        return str(self.val)


class BinaryOp(ExprOp):
    """Base class for binary arithmetic operations in TE."""

    def __init__(self, a: ExprOp | int | float, b: ExprOp | int | float) -> None:
        """Initialize binary operation operands."""
        self.a = a if isinstance(a, ExprOp) else Const(a)
        self.b = b if isinstance(b, ExprOp) else Const(b)


class Add(BinaryOp):
    """Addition operation in TE."""

    __dummy__ = True


class Sub(BinaryOp):
    """Subtraction operation in TE."""

    __dummy__ = True


class Mul(BinaryOp):
    """Multiplication operation in TE."""

    __dummy__ = True


class Div(BinaryOp):
    """Division operation in TE."""

    __dummy__ = True


class CallOp(ExprOp):
    """Intrinsic math function call in TE."""

    def __init__(self, name: str, args: list[ExprOp | int | float]) -> None:
        """Initialize call operation with function name and arguments."""
        self.name = name
        self.args = [a if isinstance(a, ExprOp) else Const(a) for a in args]


def exp(x: ExprOp | int | float) -> CallOp:
    """Compute exponential of a TE expression."""
    return CallOp("exp", [x])


def log(x: ExprOp | int | float) -> CallOp:
    """Compute natural logarithm of a TE expression."""
    return CallOp("log", [x])


def sigmoid(x: ExprOp | int | float) -> CallOp:
    """Compute sigmoid of a TE expression."""
    return CallOp("sigmoid", [x])


class ReduceOp(ExprOp):
    """Reduction operation over one or more ReduceAxis domains."""

    def __init__(self, combiner: str, source: ExprOp, axis: list[ReduceAxis]) -> None:
        """Initialize reduction operation."""
        self.combiner = combiner
        self.source = source
        self.axis = axis


def sum(source: ExprOp, axis: ReduceAxis | list[ReduceAxis]) -> ReduceOp:
    """Sum reduction over the specified axis or axes."""
    if not isinstance(axis, list):
        axis = [axis]
    return ReduceOp("sum", source, axis)


def max(source: ExprOp, axis: ReduceAxis | list[ReduceAxis]) -> ReduceOp:
    """Maximum reduction over the specified axis or axes."""
    if not isinstance(axis, list):
        axis = [axis]
    return ReduceOp("max", source, axis)


def min(source: ExprOp, axis: ReduceAxis | list[ReduceAxis]) -> ReduceOp:
    """Minimum reduction over the specified axis or axes."""
    if not isinstance(axis, list):
        axis = [axis]
    return ReduceOp("min", source, axis)


TensorOpType = NamedOpProtocol | str | None


class Tensor:
    """Represents a symbolic tensor produced by a TE operation."""

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: str,
        op: TensorOpType,
        value_index: int = 0,
    ) -> None:
        """Initialize a symbolic Tensor."""
        self.shape = shape
        self.dtype = dtype
        self.op = op
        self.value_index = value_index

    @property
    def name(self) -> str:
        """Return the name of the underlying operation."""
        if isinstance(self.op, NamedOpProtocol):
            return self.op.name
        return "unknown"

    def __call__(self, *indices: ExprOp) -> TensorComputeOp:
        """Index into the tensor at the given symbolic coordinates."""
        return TensorComputeOp(self, indices)


class TensorComputeOp(ExprOp):
    """Represents indexing into a Tensor within a compute body."""

    def __init__(self, tensor: Tensor, indices: tuple[ExprOp, ...]) -> None:
        """Initialize tensor indexing expression."""
        self.tensor = tensor
        self.indices = indices


class PlaceholderOp:
    """Represents an input placeholder operation."""

    def __init__(self, name: str, shape: tuple[int, ...], dtype: str) -> None:
        """Initialize placeholder metadata."""
        self.name = name
        self.shape = shape
        self.dtype = dtype


class ComputeOp:
    """Represents a tensor compute operation over iteration axes."""

    def __init__(
        self,
        name: str,
        tag: str,
        attrs: dict[str, AttrValue],
        axis: list[IterVar],
        body: list[ExprOp],
    ) -> None:
        """Initialize compute operation metadata and body."""
        self.name = name
        self.tag = tag
        self.attrs = attrs
        self.axis = axis
        self.body = body


def var(name: str, dtype: str = "int32") -> Var:
    """Create a symbolic scalar variable."""
    return Var(name, dtype)


def placeholder(
    shape: tuple[int, ...], dtype: str = "float32", name: str = "placeholder"
) -> Tensor:
    """Create a symbolic input placeholder tensor."""
    op = PlaceholderOp(name, shape, dtype)
    return Tensor(shape, dtype, op, 0)


class ComputeFnProtocol(Protocol):
    """Protocol for TE tensor compute functions."""

    def __call__(self, *indices: Var) -> ExprOp | list[ExprOp]:
        """Compute tensor expression at the given loop indices."""


def compute(
    shape: tuple[int, ...],
    fcompute: ComputeFnProtocol,
    name: str = "compute",
    tag: str = "",
    attrs: dict[str, AttrValue] | None = None,
) -> Tensor:
    """Construct a new tensor by computing over the given shape dimensions."""
    axis = [IterVar(f"ax{i}") for i in range(len(shape))]
    body_res = fcompute(*[Var(ax.name) for ax in axis])
    body = body_res if isinstance(body_res, list) else [body_res]
    op = ComputeOp(name, tag, attrs or {}, axis, body)
    return Tensor(shape, "float32", op, 0)


def reduce_axis(dom: tuple[int, int], name: str = "rv") -> ReduceAxis:
    """Create a reduction axis over the given domain."""
    return ReduceAxis(name, dom)
