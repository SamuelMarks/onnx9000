from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class IterVar:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"IterVar({self.name})"


class ReduceAxis(IterVar):
    def __init__(self, name: str, dom: tuple[int, int]):
        super().__init__(name)
        self.dom = dom

    def __repr__(self):
        return f"ReduceAxis({self.name}, dom={self.dom})"


class ExprOp:
    """Base class for math ops on TE variables."""

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    def __truediv__(self, other):
        return Div(self, other)

    def __rtruediv__(self, other):
        return Div(other, self)


class Var(ExprOp):
    def __init__(self, name: str, dtype: str = "int32"):
        self.name = name
        self.dtype = dtype

    def __repr__(self):
        return self.name


class Const(ExprOp):
    def __init__(self, val: Any):
        self.val = val

    def __repr__(self):
        return str(self.val)


class BinaryOp(ExprOp):
    def __init__(self, a, b):
        self.a = a if isinstance(a, ExprOp) else Const(a)
        self.b = b if isinstance(b, ExprOp) else Const(b)


class Add(BinaryOp):
    pass


class Sub(BinaryOp):
    pass


class Mul(BinaryOp):
    pass


class Div(BinaryOp):
    pass


class CallOp(ExprOp):
    def __init__(self, name: str, args: list[ExprOp]):
        self.name = name
        self.args = [a if isinstance(a, ExprOp) else Const(a) for a in args]


def exp(x):
    return CallOp("exp", [x])


def log(x):
    return CallOp("log", [x])


def sigmoid(x):
    return CallOp("sigmoid", [x])


class ReduceOp(ExprOp):
    def __init__(self, combiner: str, source: ExprOp, axis: list[ReduceAxis]):
        self.combiner = combiner
        self.source = source
        self.axis = axis


def sum(source: ExprOp, axis: Union[ReduceAxis, list[ReduceAxis]]):
    if not isinstance(axis, list):
        axis = [axis]
    return ReduceOp("sum", source, axis)


def max(source: ExprOp, axis: Union[ReduceAxis, list[ReduceAxis]]):
    if not isinstance(axis, list):
        axis = [axis]
    return ReduceOp("max", source, axis)


def min(source: ExprOp, axis: Union[ReduceAxis, list[ReduceAxis]]):
    if not isinstance(axis, list):
        axis = [axis]
    return ReduceOp("min", source, axis)


class Tensor:
    def __init__(self, shape: tuple[int, ...], dtype: str, op: Any, value_index: int = 0):
        self.shape = shape
        self.dtype = dtype
        self.op = op
        self.value_index = value_index

    def __call__(self, *indices):
        return TensorComputeOp(self, indices)


class TensorComputeOp(ExprOp):
    def __init__(self, tensor: Tensor, indices: tuple[ExprOp, ...]):
        self.tensor = tensor
        self.indices = indices


class PlaceholderOp:
    def __init__(self, name: str, shape: tuple[int, ...], dtype: str):
        self.name = name
        self.shape = shape
        self.dtype = dtype


class ComputeOp:
    def __init__(
        self, name: str, tag: str, attrs: dict[str, Any], axis: list[IterVar], body: list[ExprOp]
    ):
        self.name = name
        self.tag = tag
        self.attrs = attrs
        self.axis = axis
        self.body = body


def var(name: str, dtype: str = "int32") -> Var:
    return Var(name, dtype)


def placeholder(
    shape: tuple[int, ...], dtype: str = "float32", name: str = "placeholder"
) -> Tensor:
    op = PlaceholderOp(name, shape, dtype)
    return Tensor(shape, dtype, op, 0)


def compute(
    shape: tuple[int, ...],
    fcompute: Callable,
    name: str = "compute",
    tag: str = "",
    attrs: Optional[dict[str, Any]] = None,
) -> Tensor:
    axis = [IterVar(f"ax{i}") for i in range(len(shape))]
    body = fcompute(*[Var(ax.name) for ax in axis])
    if not isinstance(body, list):
        body = [body]
    op = ComputeOp(name, tag, attrs or {}, axis, body)
    return Tensor(shape, "float32", op, 0)


def reduce_axis(dom: tuple[int, int], name: str = "rv") -> ReduceAxis:
    return ReduceAxis(name, dom)
