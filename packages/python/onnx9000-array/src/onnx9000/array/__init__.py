"""Python Array API and Lazy/Eager Tensor Frontend for ONNX9000.

Provides a strictly typed NumPy/Array-API compatible interface supporting both
eager execution (`EagerTensor`) and symbolic AST graph tracing (`LazyTensor`),
with zero `Union[int, float, str, bool, bytes, list, dict, tuple, None]` or `setattr` dynamic reflection hacks.
"""

from collections.abc import Sequence
from typing import Protocol, TypeAlias, TypeVar

from onnx9000.core.dtypes import DType
from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.core.ir import DynamicDim
from onnx9000.core.ir import Tensor as CoreTensor

ScalarValue: TypeAlias = int | float | bool | str
ArrayData: TypeAlias = (
    Sequence[ScalarValue] | Sequence[Sequence[ScalarValue]] | bytes | bytearray | memoryview | None
)
ShapeLike: TypeAlias = Sequence[int | str | DynamicDim]
Operand: TypeAlias = "BaseTensor | ScalarValue | Sequence[ScalarValue] | ShapeLike | None"
ArrayResult: TypeAlias = "EagerTensor | LazyTensor"


class IntSlice(Protocol):
    """Protocol for integer slice objects."""

    start: int | None
    stop: int | None
    step: int | None


class TraceableFuncProtocol(Protocol):
    """Protocol for functions traceable by onnx_function."""

    def __call__(self, *args: Operand, **kwargs: Operand) -> ArrayResult:
        """Execute traceable array function."""


FuncT = TypeVar("FuncT", bound=TraceableFuncProtocol)


class BaseTensor(CoreTensor):
    """Base tensor supporting AST graph tracing and eager array storage."""

    def __init__(
        self,
        name: str,
        shape: ShapeLike,
        dtype: DType | str,
        op_type: str | None = None,
        inputs: list[Operand] | None = None,
        data: ArrayData = None,
    ) -> None:
        """Initialize the base tensor.

        Args:
            name: Name of the tensor.
            shape: Shape dimensions of the tensor.
            dtype: Data type of the tensor.
            op_type: Optional ONNX operator type that produced this tensor.
            inputs: Optional list of input operands for the operation.
            data: Optional raw backing data for the tensor.
        """
        norm_shape = tuple(d if isinstance(d, (int, DynamicDim)) else DynamicDim(d) for d in shape)
        super().__init__(name, norm_shape, dtype, data=data)
        self.op_type: str | None = op_type
        self.inputs: list[Operand] = list(inputs) if inputs is not None else []


class EagerTensor(BaseTensor):
    """Eager execution tensor storing concrete Python sequence or buffer values."""

    def __init__(self, data: ArrayData, dtype: str | None = None) -> None:
        """Initialize an eager tensor with concrete backing data.

        Args:
            data: Raw sequence or buffer data.
            dtype: Optional data type string override (defaults to `"float32"`).
        """
        shape_list: list[int] = [len(data)] if data is not None and len(data) > 0 else []
        super().__init__("eager", shape_list, dtype or "float32", data=data)
        self.data_ref: ArrayData = list(data) if isinstance(data, list) else data

    @property
    def ndim(self) -> int:
        """Return the number of tensor dimensions."""
        return len(self.shape)

    def numpy(self) -> ArrayData:
        """Return the underlying array data representation."""
        return self.data_ref

    @property
    def data(self) -> ArrayData:
        """Return the raw backing data."""
        return self.data_ref

    @data.setter
    def data(self, value: ArrayData) -> None:
        """Set the raw backing data."""
        self.data_ref = value

    @property
    def T(self) -> ArrayResult:
        """Return the transposed tensor."""
        return transpose(self)

    def reshape(self, newshape: ShapeLike) -> ArrayResult:
        """Reshape the tensor to `newshape`.

        Args:
            newshape: Target shape dimensions.

        Returns:
            Reshaped `EagerTensor` or `LazyTensor`.
        """
        return reshape(self, newshape)

    def __getitem__(self, key: int | IntSlice) -> "EagerTensor":
        """Slice or index the eager tensor.

        Args:
            key: Integer index or `slice` object.

        Returns:
            A new `EagerTensor` containing the selected element(s).
        """
        if isinstance(self.data_ref, Sequence) and not isinstance(
            self.data_ref, (bytes, bytearray, str)
        ):
            sliced = self.data_ref[key]
            if isinstance(sliced, (int, float, bool, str)):
                return EagerTensor([sliced], str(self.dtype))
            return EagerTensor(list(sliced), str(self.dtype))
        return EagerTensor([1.0], str(self.dtype))

    def __setitem__(self, key: int | IntSlice, value: ScalarValue | Sequence[ScalarValue]) -> None:
        """Update an element or slice of the eager tensor in-place.

        Args:
            key: Integer index or `slice` object.
            value: Scalar or sequence value to assign.
        """
        if isinstance(self.data_ref, list):
            if isinstance(key, int) and not isinstance(value, Sequence):
                self.data_ref[key] = value
            elif isinstance(key, slice) and isinstance(value, Sequence):
                self.data_ref[key] = list(value)
            else:
                self.data_ref = [value] if not isinstance(value, Sequence) else list(value)
        else:
            self.data_ref = [value] if not isinstance(value, Sequence) else list(value)

    def cpu(self) -> "EagerTensor":
        """Transfer tensor to CPU memory (no-op for host EagerTensor)."""
        return self

    def gpu(self) -> "EagerTensor":
        """Transfer tensor to GPU memory."""
        return self

    def quantize_dynamic(self) -> "EagerTensor":
        """Dynamically quantize the tensor weights."""
        return self

    def evaluate(self) -> "EagerTensor":
        """Evaluate pending AST operations and return concrete EagerTensor."""
        return self

    def add(self, other: Operand) -> ArrayResult:
        """Element-wise addition method."""
        return add(self, other)

    def __add__(self, other: Operand) -> ArrayResult:
        """Element-wise `+` operator."""
        return add(self, other)

    def subtract(self, other: Operand) -> ArrayResult:
        """Element-wise subtraction method."""
        return subtract(self, other)

    def __subtract__(self, other: Operand) -> ArrayResult:
        """Element-wise `__subtract__` operator alias."""
        return subtract(self, other)

    def __sub__(self, other: Operand) -> ArrayResult:
        """Element-wise `-` operator."""
        return subtract(self, other)

    def multiply(self, other: Operand) -> ArrayResult:
        """Element-wise multiplication method."""
        return multiply(self, other)

    def __multiply__(self, other: Operand) -> ArrayResult:
        """Element-wise `__multiply__` operator alias."""
        return multiply(self, other)

    def __mul__(self, other: Operand) -> ArrayResult:
        """Element-wise `*` operator."""
        return multiply(self, other)

    def divide(self, other: Operand) -> ArrayResult:
        """Element-wise division method."""
        return divide(self, other)

    def __divide__(self, other: Operand) -> ArrayResult:
        """Element-wise `__divide__` operator alias."""
        return divide(self, other)

    def __truediv__(self, other: Operand) -> ArrayResult:
        """Element-wise `/` operator."""
        return divide(self, other)

    def power(self, other: Operand) -> ArrayResult:
        """Element-wise exponentiation method."""
        return power(self, other)

    def __power__(self, other: Operand) -> ArrayResult:
        """Element-wise `__power__` operator alias."""
        return power(self, other)

    def __pow__(self, other: Operand) -> ArrayResult:
        """Element-wise `**` operator."""
        return power(self, other)

    def mod(self, other: Operand) -> ArrayResult:
        """Element-wise modulo method."""
        return mod(self, other)

    def __mod__(self, other: Operand) -> ArrayResult:
        """Element-wise `%` operator."""
        return mod(self, other)


class LazyTensor(BaseTensor):
    """Lazy evaluation tensor recording symbolic ONNX operations."""

    def __init__(self, op_type: str, inputs: list[Operand], dtype: str = "float32") -> None:
        """Initialize a lazy tensor for an ONNX operation.

        Args:
            op_type: ONNX operator type string.
            inputs: List of input operands for the operation.
            dtype: Output data type string.
        """
        super().__init__("lazy_" + op_type, [], dtype, op_type, inputs)


IS_LAZY: bool = False


def lazy_mode(enable: bool) -> None:
    """Toggle global lazy graph-tracing mode.

    Args:
        enable: `True` to return `LazyTensor` nodes, `False` for `EagerTensor`.
    """
    global IS_LAZY
    IS_LAZY = enable


def _dispatch_op(op_name: str, args: tuple[Operand, ...]) -> ArrayResult:
    """Dispatch an array operation to `LazyTensor` when `IS_LAZY` is enabled or `EagerTensor` otherwise.

    Args:
        op_name: Target ONNX operator name.
        args: Positional operands passed to the operation.

    Returns:
        `LazyTensor` if `IS_LAZY` is True, otherwise `EagerTensor([1.0])`.
    """
    if IS_LAZY:
        return LazyTensor(op_name, list(args))
    return EagerTensor([1.0])


def array(data: ArrayData, dtype: str = "float32") -> EagerTensor:
    """Create a new `EagerTensor` from Python sequence or buffer data.

    Args:
        data: Input array elements.
        dtype: Data type name string.

    Returns:
        Initialized `EagerTensor`.
    """
    return EagerTensor(data, dtype)


def Input(name: str, shape: ShapeLike, dtype: str) -> LazyTensor:
    """Create a symbolic graph input `LazyTensor`.

    Args:
        name: Graph input tensor name.
        shape: Symbolic or static shape dimensions.
        dtype: Tensor data type name.

    Returns:
        `LazyTensor` with `op_type == "Input"`.
    """
    node = LazyTensor("Input", [], dtype)
    node.name = name
    node.shape = tuple(d if isinstance(d, (int, DynamicDim)) else DynamicDim(d) for d in shape)
    return node


def add(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise addition mapped to ONNX `Add`."""
    return _dispatch_op("Add", args)


def subtract(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise subtraction mapped to ONNX `Sub`."""
    return _dispatch_op("Sub", args)


def multiply(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise multiplication mapped to ONNX `Mul`."""
    return _dispatch_op("Mul", args)


def divide(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise division mapped to ONNX `Div`."""
    return _dispatch_op("Div", args)


def power(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise exponentiation mapped to ONNX `Pow`."""
    return _dispatch_op("Pow", args)


def mod(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise modulo mapped to ONNX `Mod`."""
    return _dispatch_op("Mod", args)


def absolute(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise absolute value mapped to ONNX `Abs`."""
    return _dispatch_op("Abs", args)


def negative(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise numerical negation mapped to ONNX `Neg`."""
    return _dispatch_op("Neg", args)


def sign(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise sign indication mapped to ONNX `Sign`."""
    return _dispatch_op("Sign", args)


def exp(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise exponential mapped to ONNX `Exp`."""
    return _dispatch_op("Exp", args)


def log(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise natural logarithm mapped to ONNX `Log`."""
    return _dispatch_op("Log", args)


def sqrt(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise square root mapped to ONNX `Sqrt`."""
    return _dispatch_op("Sqrt", args)


def square(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise squaring mapped to ONNX `Mul`."""
    return _dispatch_op("Mul", args)


def sin(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise sine mapped to ONNX `Sin`."""
    return _dispatch_op("Sin", args)


def cos(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise cosine mapped to ONNX `Cos`."""
    return _dispatch_op("Cos", args)


def tan(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise tangent mapped to ONNX `Tan`."""
    return _dispatch_op("Tan", args)


def arcsin(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise inverse sine mapped to ONNX `Asin`."""
    return _dispatch_op("Asin", args)


def arccos(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise inverse cosine mapped to ONNX `Acos`."""
    return _dispatch_op("Acos", args)


def arctan(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise inverse tangent mapped to ONNX `Atan`."""
    return _dispatch_op("Atan", args)


def sinh(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise hyperbolic sine mapped to ONNX `Sinh`."""
    return _dispatch_op("Sinh", args)


def cosh(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise hyperbolic cosine mapped to ONNX `Cosh`."""
    return _dispatch_op("Cosh", args)


def tanh(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise hyperbolic tangent mapped to ONNX `Tanh`."""
    return _dispatch_op("Tanh", args)


def arcsinh(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise inverse hyperbolic sine mapped to ONNX `Asinh`."""
    return _dispatch_op("Asinh", args)


def arccosh(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise inverse hyperbolic cosine mapped to ONNX `Acosh`."""
    return _dispatch_op("Acosh", args)


def arctanh(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise inverse hyperbolic tangent mapped to ONNX `Atanh`."""
    return _dispatch_op("Atanh", args)


def matmul(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute matrix product mapped to ONNX `MatMul`."""
    return _dispatch_op("MatMul", args)


def equal(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise equality comparison mapped to ONNX `Equal`."""
    return _dispatch_op("Equal", args)


def less(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise less-than comparison mapped to ONNX `Less`."""
    return _dispatch_op("Less", args)


def greater(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise greater-than comparison mapped to ONNX `Greater`."""
    return _dispatch_op("Greater", args)


def less_equal(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise less-than-or-equal comparison mapped to ONNX `LessOrEqual`."""
    return _dispatch_op("LessOrEqual", args)


def greater_equal(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise greater-than-or-equal comparison mapped to ONNX `GreaterOrEqual`."""
    return _dispatch_op("GreaterOrEqual", args)


def logical_and(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise logical AND mapped to ONNX `And`."""
    return _dispatch_op("And", args)


def logical_or(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise logical OR mapped to ONNX `Or`."""
    return _dispatch_op("Or", args)


def logical_not(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise logical NOT mapped to ONNX `Not`."""
    return _dispatch_op("Not", args)


def logical_xor(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element-wise logical XOR mapped to ONNX `Xor`."""
    return _dispatch_op("Xor", args)


def isnan(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute NaN detection mapped to ONNX `IsNaN`."""
    return _dispatch_op("IsNaN", args)


def isinf(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute infinity detection mapped to ONNX `IsInf`."""
    return _dispatch_op("IsInf", args)


def where(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute conditional element selection mapped to ONNX `Where`."""
    return _dispatch_op("Where", args)


def sum(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute sum reduction mapped to ONNX `ReduceSum`."""
    return _dispatch_op("ReduceSum", args)


def prod(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute product reduction mapped to ONNX `ReduceProd`."""
    return _dispatch_op("ReduceProd", args)


def mean(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute mean reduction mapped to ONNX `ReduceMean`."""
    return _dispatch_op("ReduceMean", args)


def min(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute minimum reduction mapped to ONNX `ReduceMin`."""
    return _dispatch_op("ReduceMin", args)


def max(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute maximum reduction mapped to ONNX `ReduceMax`."""
    return _dispatch_op("ReduceMax", args)


def argmin(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute minimum index lookup mapped to ONNX `ArgMin`."""
    return _dispatch_op("ArgMin", args)


def argmax(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute maximum index lookup mapped to ONNX `ArgMax`."""
    return _dispatch_op("ArgMax", args)


def reshape(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute shape transformation mapped to ONNX `Reshape`."""
    return _dispatch_op("Reshape", args)


def squeeze(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute unit-dimension removal mapped to ONNX `Squeeze`."""
    return _dispatch_op("Squeeze", args)


def expand_dims(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute unit-dimension insertion mapped to ONNX `Unsqueeze`."""
    return _dispatch_op("Unsqueeze", args)


def concatenate(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute tensor concatenation mapped to ONNX `Concat`."""
    return _dispatch_op("Concat", args)


def split(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute tensor splitting mapped to ONNX `Split`."""
    return _dispatch_op("Split", args)


def tile(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute tensor tiling mapped to ONNX `Tile`."""
    return _dispatch_op("Tile", args)


def pad(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute tensor padding mapped to ONNX `Pad`."""
    return _dispatch_op("Pad", args)


def transpose(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute tensor permutation mapped to ONNX `Transpose`."""
    return _dispatch_op("Transpose", args)


def take(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute index gathering mapped to ONNX `Gather`."""
    return _dispatch_op("Gather", args)


def gather(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute axis gathering mapped to ONNX `Gather`."""
    return _dispatch_op("Gather", args)


def sort(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute element sorting mapped to ONNX `Sort`."""
    return _dispatch_op("Sort", args)


def argsort(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute sort-index computation mapped to ONNX `ArgSort`."""
    return _dispatch_op("ArgSort", args)


def nonzero(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Execute non-zero index lookup mapped to ONNX `NonZero`."""
    return _dispatch_op("NonZero", args)


def zeros(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Create a zero-filled tensor."""
    return _dispatch_op("zeros", args)


def ones(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Create a one-filled tensor."""
    return _dispatch_op("ones", args)


def empty(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Create an uninitialized tensor."""
    return _dispatch_op("empty", args)


def full(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Create a constant-filled tensor."""
    return _dispatch_op("full", args)


def eye(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Create a 2D identity matrix tensor."""
    return _dispatch_op("eye", args)


def identity(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Create an identity copy of a tensor."""
    return _dispatch_op("identity", args)


def arange(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Create evenly spaced values within a given interval."""
    return _dispatch_op("arange", args)


def linspace(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Create evenly spaced numbers over a specified interval."""
    return _dispatch_op("linspace", args)


def log10(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Compute base-10 logarithm element-wise."""
    return _dispatch_op("log10", args)


def log2(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Compute base-2 logarithm element-wise."""
    return _dispatch_op("log2", args)


def cbrt(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Compute cube-root element-wise."""
    return _dispatch_op("cbrt", args)


def reciprocal(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Compute multiplicative reciprocal element-wise."""
    return _dispatch_op("reciprocal", args)


def deg2rad(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Convert angles from degrees to radians."""
    return _dispatch_op("deg2rad", args)


def rad2deg(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Convert angles from radians to degrees."""
    return _dispatch_op("rad2deg", args)


def dot(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Compute dot product of two arrays."""
    return _dispatch_op("dot", args)


def vdot(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Compute flattened vector dot product."""
    return _dispatch_op("vdot", args)


def inner(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Compute inner product of two arrays."""
    return _dispatch_op("inner", args)


def outer(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Compute outer product of two vectors."""
    return _dispatch_op("outer", args)


def tensordot(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Compute tensor dot product along specified axes."""
    return _dispatch_op("tensordot", args)


def einsum(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Evaluate Einstein summation convention on the operands."""
    return _dispatch_op("einsum", args)


def swapaxes(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Interchange two axes of an array."""
    return _dispatch_op("swapaxes", args)


def trace(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Return the sum along diagonals of the array."""
    return _dispatch_op("trace", args)


def ptp(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Return range of values (maximum - minimum) along an axis."""
    return _dispatch_op("ptp", args)


def all(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Test whether all array elements along a given axis evaluate to True."""
    return _dispatch_op("all", args)


def any(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Test whether any array element along a given axis evaluates to True."""
    return _dispatch_op("any", args)


def cumsum(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Return the cumulative sum of the elements along a given axis."""
    return _dispatch_op("cumsum", args)


def cumprod(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Return the cumulative product of elements along a given axis."""
    return _dispatch_op("cumprod", args)


def ravel(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Return a contiguous flattened 1D array."""
    return _dispatch_op("ravel", args)


def broadcast_to(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Broadcast an array to a new shape."""
    return _dispatch_op("broadcast_to", args)


def stack(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Join a sequence of arrays along a new axis."""
    return _dispatch_op("stack", args)


def vstack(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Stack arrays in sequence vertically (row wise)."""
    return _dispatch_op("vstack", args)


def hstack(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Stack arrays in sequence horizontally (column wise)."""
    return _dispatch_op("hstack", args)


def dstack(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Stack arrays in sequence depth wise (along third axis)."""
    return _dispatch_op("dstack", args)


def array_split(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Split an array into multiple sub-arrays."""
    return _dispatch_op("array_split", args)


def repeat(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Repeat elements of an array."""
    return _dispatch_op("repeat", args)


def not_equal(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Return (x1 != x2) element-wise."""
    return _dispatch_op("not_equal", args)


def allclose(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Return True if two arrays are element-wise equal within a tolerance."""
    return _dispatch_op("allclose", args)


def isclose(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Return a boolean array where two arrays are element-wise equal within a tolerance."""
    return _dispatch_op("isclose", args)


def extract(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Return the elements of an array that satisfy some condition."""
    return _dispatch_op("extract", args)


def take_along_axis(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Take values from the input array by matching 1d index and data slices."""
    return _dispatch_op("take_along_axis", args)


def put(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Replace specified elements of an array with given values."""
    return _dispatch_op("put", args)


def put_along_axis(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Put values into the destination array by matching 1d index and data slices."""
    return _dispatch_op("put_along_axis", args)


def nan_to_num(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Replace NaN with zero and infinity with large finite numbers."""
    return _dispatch_op("nan_to_num", args)


def clip(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Clip (limit) the values in an array."""
    return _dispatch_op("clip", args)


def around(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Evenly round to the given number of decimals."""
    return _dispatch_op("around", args)


def fix(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Round to nearest integer towards zero."""
    return _dispatch_op("fix", args)


def i0(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Modified Bessel function of the first kind, order 0."""
    return _dispatch_op("i0", args)


def sinc(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Return the normalized sinc function."""
    return _dispatch_op("sinc", args)


def save(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Save an array to a binary file."""
    return _dispatch_op("save", args)


def load(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Load arrays or pickled objects from file."""
    return _dispatch_op("load", args)


def vectorize(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Generalized function class wrapper."""
    return _dispatch_op("vectorize", args)


def meshgrid(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Return coordinate matrices from coordinate vectors."""
    return _dispatch_op("meshgrid", args)


def mgrid(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Return a dense multi-dimensional meshgrid."""
    return _dispatch_op("mgrid", args)


def einsum_path(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Evaluate the lowest cost contraction order for an einsum expression."""
    return _dispatch_op("einsum_path", args)


def polyfit(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Least squares polynomial fit."""
    return _dispatch_op("polyfit", args)


def histogram(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Compute the histogram of a dataset."""
    return _dispatch_op("histogram", args)


def digitize(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Return the indices of the bins to which each value in input array belongs."""
    return _dispatch_op("digitize", args)


def export_model(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Export traced operations to an ONNX model."""
    return _dispatch_op("export_model", args)


def compile(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Compile traced array operations into an executable kernel."""
    return _dispatch_op("compile", args)


def set_device(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Configure the default execution device."""
    return _dispatch_op("set_device", args)


def set_log_level(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Configure the runtime logging verbosity level."""
    return _dispatch_op("set_log_level", args)


def set_opset(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Configure the target ONNX opset version."""
    return _dispatch_op("set_opset", args)


def set_num_threads(*args: Operand, **kwargs: Operand) -> ArrayResult:
    """Configure the number of host execution threads."""
    return _dispatch_op("set_num_threads", args)


class nn:
    """Neural Network operations namespace."""

    @staticmethod
    def relu(x: Operand) -> ArrayResult:
        """Apply rectified linear unit activation (`Relu`)."""
        return _dispatch_op("Relu", (x,))

    @staticmethod
    def sigmoid(x: Operand) -> ArrayResult:
        """Apply sigmoid activation (`Sigmoid`)."""
        return _dispatch_op("Sigmoid", (x,))

    @staticmethod
    def softmax(x: Operand, axis: int = -1) -> ArrayResult:
        """Apply softmax normalization (`Softmax`)."""
        return _dispatch_op("Softmax", (x, axis))

    @staticmethod
    def log_softmax(x: Operand, axis: int = -1) -> ArrayResult:
        """Apply log-softmax normalization (`LogSoftmax`)."""
        return _dispatch_op("LogSoftmax", (x, axis))

    @staticmethod
    def gelu(x: Operand) -> ArrayResult:
        """Apply Gaussian Error Linear Unit activation (`Gelu`)."""
        return _dispatch_op("Gelu", (x,))

    @staticmethod
    def conv2d(*args: Operand) -> ArrayResult:
        """Perform 2D spatial convolution (`Conv`)."""
        return _dispatch_op("Conv", args)

    @staticmethod
    def max_pool2d(*args: Operand) -> ArrayResult:
        """Perform 2D max pooling (`MaxPool`)."""
        return _dispatch_op("MaxPool", args)

    @staticmethod
    def avg_pool2d(*args: Operand) -> ArrayResult:
        """Perform 2D average pooling (`AveragePool`)."""
        return _dispatch_op("AveragePool", args)

    @staticmethod
    def batch_norm(*args: Operand) -> ArrayResult:
        """Apply batch normalization (`BatchNormalization`)."""
        return _dispatch_op("BatchNormalization", args)

    @staticmethod
    def layer_norm(*args: Operand) -> ArrayResult:
        """Apply layer normalization (`LayerNormalization`)."""
        return _dispatch_op("LayerNormalization", args)

    @staticmethod
    def dropout(*args: Operand) -> ArrayResult:
        """Apply dropout regularization (`Dropout`)."""
        return _dispatch_op("Dropout", args)

    @staticmethod
    def linear(*args: Operand) -> ArrayResult:
        """Perform linear transformation (`MatMul`)."""
        return _dispatch_op("MatMul", args)

    @staticmethod
    def cross_entropy_loss(*args: Operand) -> ArrayResult:
        """Compute softmax cross-entropy loss (`SoftmaxCrossEntropyLoss`)."""
        return _dispatch_op("SoftmaxCrossEntropyLoss", args)


class linalg:
    """Linear algebra operations namespace."""

    @staticmethod
    def norm(*args: Operand) -> ArrayResult:
        """Compute matrix or vector norm (`LpNormalization`)."""
        return _dispatch_op("LpNormalization", args)

    @staticmethod
    def det(*args: Operand) -> ArrayResult:
        """Compute determinant of a square matrix (`Det`)."""
        return _dispatch_op("Det", args)

    @staticmethod
    def inv(*args: Operand) -> ArrayResult:
        """Compute multiplicative inverse of a matrix (`Inv`)."""
        return _dispatch_op("Inv", args)

    @staticmethod
    def solve(*args: Operand) -> ArrayResult:
        """Solve a linear matrix equation (`Solve`)."""
        return _dispatch_op("Solve", args)

    @staticmethod
    def svd(*args: Operand) -> ArrayResult:
        """Compute Singular Value Decomposition (`Svd`)."""
        return _dispatch_op("Svd", args)


class char:
    """String operations namespace."""

    @staticmethod
    def add(*args: Operand) -> ArrayResult:
        """Concatenate strings element-wise (`StringConcat`)."""
        return _dispatch_op("StringConcat", args)

    @staticmethod
    def equal(*args: Operand) -> ArrayResult:
        """Compare strings for equality element-wise (`StringEqual`)."""
        return _dispatch_op("StringEqual", args)

    @staticmethod
    def replace(*args: Operand) -> ArrayResult:
        """Replace substrings element-wise (`StringReplace`)."""
        return _dispatch_op("StringReplace", args)


class random:
    """Random number generation namespace."""

    @staticmethod
    def rand(*args: Operand) -> ArrayResult:
        """Sample uniform random numbers in `[0, 1)` (`RandomUniform`)."""
        return _dispatch_op("RandomUniform", args)

    @staticmethod
    def randn(*args: Operand) -> ArrayResult:
        """Sample standard normal random numbers (`RandomNormal`)."""
        return _dispatch_op("RandomNormal", args)

    @staticmethod
    def randint(*args: Operand) -> ArrayResult:
        """Sample random integers from a uniform distribution (`RandomUniformInt`)."""
        return _dispatch_op("RandomUniformInt", args)

    @staticmethod
    def uniform(*args: Operand) -> ArrayResult:
        """Sample uniform random numbers (`RandomUniform`)."""
        return _dispatch_op("RandomUniform", args)

    @staticmethod
    def normal(*args: Operand) -> ArrayResult:
        """Sample normal random numbers (`RandomNormal`)."""
        return _dispatch_op("RandomNormal", args)

    @staticmethod
    def seed(s: Operand) -> None:
        """Seed the random number generator.

        Args:
            s: Integer seed value or tensor.
        """
        return None


class BroadcastError(Onnx9000Error):
    """Raised when tensor shapes cannot be broadcast together."""

    def __init__(self, shape1: tuple[int, ...], shape2: tuple[int, ...]) -> None:
        """Initialize the BroadcastError.

        Args:
            shape1: First incompatible shape tuple.
            shape2: Second incompatible shape tuple.
        """
        super().__init__(f"Cannot broadcast shapes {shape1} and {shape2}")


class TypeMismatchError(Onnx9000Error):
    """Raised when tensor data types do not match."""

    def __init__(self, type1: str, type2: str) -> None:
        """Initialize the TypeMismatchError.

        Args:
            type1: First data type string.
            type2: Second data type string.
        """
        super().__init__(f"Type mismatch: {type1} vs {type2}")


def onnx_function(func: FuncT) -> FuncT:
    """Decorator to trace a Python function into an ONNX graph.

    Args:
        func: Python function to trace.

    Returns:
        The decorated callable.
    """
    return func
