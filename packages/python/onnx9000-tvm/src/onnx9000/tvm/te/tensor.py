"""TVM submodule for AST and optimization."""

from typing import Any, Callable, Optional, Union
from ..tir.expr import Expr


class ExprOp:
    """Base class for math ops on TE variables."""

    def __add__(self, other):
        """Evaluate or manipulates TVM AST nodes."""
        return Add(self, other)

    def __radd__(self, other):
        """Evaluate or manipulates TVM AST nodes."""
        return Add(other, self)

    def __sub__(self, other):
        """Evaluate or manipulates TVM AST nodes."""
        return Sub(self, other)

    def __rsub__(self, other):
        """Evaluate or manipulates TVM AST nodes."""
        return Sub(other, self)

    def __mul__(self, other):
        """Evaluate or manipulates TVM AST nodes."""
        return Mul(self, other)

    def __rmul__(self, other):
        """Evaluate or manipulates TVM AST nodes."""
        return Mul(other, self)

    def __truediv__(self, other):
        """Evaluate or manipulates TVM AST nodes."""
        return Div(self, other)

    def __rtruediv__(self, other):
        """Evaluate or manipulates TVM AST nodes."""
        return Div(other, self)


class IterVar(ExprOp):
    """Evaluates or manipulates TVM AST nodes."""

    def __init__(self, name: str):
        """Evaluate or manipulates TVM AST nodes."""
        self.name = name
        self.dtype = "int32"

    def __repr__(self):
        """Evaluate or manipulates TVM AST nodes."""
        return f"IterVar({self.name})"


class ReduceAxis(IterVar):
    """Evaluates or manipulates TVM AST nodes."""

    def __init__(self, name: str, dom: tuple[int, int]):
        """Evaluate or manipulates TVM AST nodes."""
        super().__init__(name)
        self.dom = dom

    def __repr__(self):
        """Evaluate or manipulates TVM AST nodes."""
        return f"ReduceAxis({self.name}, dom={self.dom})"


class Var(ExprOp):
    """Evaluates or manipulates TVM AST nodes."""

    def __init__(self, name: str, dtype: str = "int32"):
        """Evaluate or manipulates TVM AST nodes."""
        self.name = name
        self.dtype = dtype

    def __repr__(self):
        """Evaluate or manipulates TVM AST nodes."""
        return self.name


class Const(ExprOp):
    """Evaluates or manipulates TVM AST nodes."""

    def __init__(self, val: Any):
        """Evaluate or manipulates TVM AST nodes."""
        self.val = val

    def __repr__(self):
        """Evaluate or manipulates TVM AST nodes."""
        return str(self.val)


class BinaryOp(ExprOp):
    """Evaluates or manipulates TVM AST nodes."""

    def __init__(self, a, b):
        """Evaluate or manipulates TVM AST nodes."""
        self.a = a if isinstance(a, ExprOp) else Const(a)
        self.b = b if isinstance(b, ExprOp) else Const(b)


class Add(BinaryOp):
    """Evaluates or manipulates TVM AST nodes."""

    pass


class Sub(BinaryOp):
    """Evaluates or manipulates TVM AST nodes."""

    pass


class Mul(BinaryOp):
    """Evaluates or manipulates TVM AST nodes."""

    pass


class Div(BinaryOp):
    """Evaluates or manipulates TVM AST nodes."""

    pass


class CallOp(ExprOp):
    """Evaluates or manipulates TVM AST nodes."""

    def __init__(self, name: str, args: list[ExprOp]):
        """Evaluate or manipulates TVM AST nodes."""
        self.name = name
        self.args = [a if isinstance(a, ExprOp) else Const(a) for a in args]


def exp(x):
    """Evaluate or manipulates TVM AST nodes."""
    return CallOp("exp", [x])


def log(x):
    """Evaluate or manipulates TVM AST nodes."""
    return CallOp("log", [x])


def sigmoid(x):
    """Evaluate or manipulates TVM AST nodes."""
    return CallOp("sigmoid", [x])


class ReduceOp(ExprOp):
    """Evaluates or manipulates TVM AST nodes."""

    def __init__(self, combiner: str, source: ExprOp, axis: list[ReduceAxis]):
        """Evaluate or manipulates TVM AST nodes."""
        self.combiner = combiner
        self.source = source
        self.axis = axis


def sum(source: ExprOp, axis: Union[ReduceAxis, list[ReduceAxis]]):
    """Evaluate or manipulates TVM AST nodes."""
    if not isinstance(axis, list):
        axis = [axis]
    return ReduceOp("sum", source, axis)


def max(source: ExprOp, axis: Union[ReduceAxis, list[ReduceAxis]]):
    """Evaluate or manipulates TVM AST nodes."""
    if not isinstance(axis, list):
        axis = [axis]
    return ReduceOp("max", source, axis)


def min(source: ExprOp, axis: Union[ReduceAxis, list[ReduceAxis]]):
    """Evaluate or manipulates TVM AST nodes."""
    if not isinstance(axis, list):
        axis = [axis]
    return ReduceOp("min", source, axis)


class Tensor:
    """Evaluates or manipulates TVM AST nodes."""

    def __init__(self, shape: tuple[int, ...], dtype: str, op: Any, value_index: int = 0):
        """Evaluate or manipulates TVM AST nodes."""
        self.shape = shape
        self.dtype = dtype
        self.op = op
        self.value_index = value_index

    @property
    def name(self) -> str:
        """Name."""
        return getattr(self.op, "name", "unknown")

    def __call__(self, *indices):
        """Evaluate or manipulates TVM AST nodes."""
        return TensorComputeOp(self, indices)


class TensorComputeOp(ExprOp):
    """Evaluates or manipulates TVM AST nodes."""

    def __init__(self, tensor: Tensor, indices: tuple[ExprOp, ...]):
        """Evaluate or manipulates TVM AST nodes."""
        self.tensor = tensor
        self.indices = indices


class PlaceholderOp:
    """Evaluates or manipulates TVM AST nodes."""

    def __init__(self, name: str, shape: tuple[int, ...], dtype: str):
        """Evaluate or manipulates TVM AST nodes."""
        self.name = name
        self.shape = shape
        self.dtype = dtype


class ComputeOp:
    """Evaluates or manipulates TVM AST nodes."""

    def __init__(
        self, name: str, tag: str, attrs: dict[str, Any], axis: list[IterVar], body: list[ExprOp]
    ):
        """Evaluate or manipulates TVM AST nodes."""
        self.name = name
        self.tag = tag
        self.attrs = attrs
        self.axis = axis
        self.body = body


def var(name: str, dtype: str = "int32") -> Var:
    """Evaluate or manipulates TVM AST nodes."""
    return Var(name, dtype)


def placeholder(
    shape: tuple[int, ...], dtype: str = "float32", name: str = "placeholder"
) -> Tensor:
    """Evaluate or manipulates TVM AST nodes."""
    op = PlaceholderOp(name, shape, dtype)
    return Tensor(shape, dtype, op, 0)


def compute(
    shape: tuple[int, ...],
    fcompute: Callable,
    name: str = "compute",
    tag: str = "",
    attrs: Optional[dict[str, Any]] = None,
) -> Tensor:
    """Evaluate or manipulates TVM AST nodes."""
    axis = [IterVar(f"ax{i}") for i in range(len(shape))]
    body = fcompute(*[Var(ax.name) for ax in axis])
    if not isinstance(body, list):
        body = [body]
    op = ComputeOp(name, tag, attrs or {}, axis, body)
    return Tensor(shape, "float32", op, 0)


def reduce_axis(dom: tuple[int, int], name: str = "rv") -> ReduceAxis:
    """Evaluate or manipulates TVM AST nodes."""
    return ReduceAxis(name, dom)
