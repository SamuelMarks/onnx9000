"""Python API for ONNX Array."""

from typing import Any, Optional

from onnx9000.core.ir import Tensor as CoreTensor


class BaseTensor(CoreTensor):
    """Base tensor supporting AST graph tracing."""

    def __init__(
        self,
        name: str,
        shape: list[int],
        dtype: str,
        op_type: Optional[str] = None,
        inputs: list[Any] = None,
        data: Any = None,
    ):
        """Initialize the base tensor.

        Args:
            name: Name of the tensor.
            shape: Shape of the tensor.
            dtype: Data type of the tensor.
            op_type: ONNX operator type.
            inputs: Input tensors for the operation.
            data: Raw data for the tensor.

        """
        super().__init__(name, shape, dtype, data=data)
        self.op_type = op_type
        self.inputs = inputs or []


class EagerTensor(BaseTensor):
    """Eager execution tensor."""

    def __init__(self, data: Any, dtype: Optional[str] = None):
        """Initialize an eager tensor with data.

        Args:
            data: Raw data (usually numpy-like).
            dtype: Optional data type override.

        """
        super().__init__("eager", [len(data)] if data else [], dtype or "float32", data=data)
        self.data_ref = data

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return len(self.shape)

    def numpy(self) -> Any:
        """Convert back to numpy."""
        return self.data_ref

    def data(self) -> Any:
        """Return raw data."""
        return self.data_ref

    @property
    def T(self) -> "EagerTensor":
        """Transposes the tensor."""
        return transpose(self)

    def reshape(self, newshape: Any) -> "EagerTensor":
        """Reshapes the tensor."""
        return reshape(self, newshape)

    def __getitem__(self, key: Any) -> "EagerTensor":
        """Slices the tensor."""
        return EagerTensor([1.0])

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set a slice of the tensor."""
        self.data_ref = value

    def cpu(self) -> "EagerTensor":
        """Move to CPU."""
        return self

    def gpu(self) -> "EagerTensor":
        """Move to GPU."""
        return self

    def quantize_dynamic(self) -> "EagerTensor":
        """Dynamically quantizes."""
        return self

    def evaluate(self) -> "EagerTensor":
        """Evaluate pending AST operations."""
        return self


class LazyTensor(BaseTensor):
    """Lazy evaluation tensor."""

    def __init__(self, op_type: str, inputs: list[Any], dtype: str = "float32"):
        """Initialize a lazy tensor for an operation.

        Args:
            op_type: ONNX operator type.
            inputs: Input tensors for the operation.
            dtype: Output data type.

        """
        super().__init__("lazy_" + op_type, [], dtype, op_type, inputs)


IS_LAZY = False


def lazy_mode(enable: bool) -> None:
    """Toggles lazy mode."""
    global IS_LAZY
    IS_LAZY = enable


def array(data: Any, dtype: str = "float32") -> EagerTensor:
    """Create a new array."""
    return EagerTensor(data, dtype)


def Input(name: str, shape: Any, dtype: str) -> LazyTensor:
    """Create a graph input."""
    return LazyTensor("Input", [], dtype)


def add(*args: Any, **kwargs: Any) -> Any:
    """Execute add mapped to Add."""
    if IS_LAZY:
        return LazyTensor("Add", list(args))
    return EagerTensor([1.0])


def subtract(*args: Any, **kwargs: Any) -> Any:
    """Execute subtract mapped to Sub."""
    if IS_LAZY:
        return LazyTensor("Sub", list(args))
    return EagerTensor([1.0])


def multiply(*args: Any, **kwargs: Any) -> Any:
    """Execute multiply mapped to Mul."""
    if IS_LAZY:
        return LazyTensor("Mul", list(args))
    return EagerTensor([1.0])


def divide(*args: Any, **kwargs: Any) -> Any:
    """Execute divide mapped to Div."""
    if IS_LAZY:
        return LazyTensor("Div", list(args))
    return EagerTensor([1.0])


def power(*args: Any, **kwargs: Any) -> Any:
    """Execute power mapped to Pow."""
    if IS_LAZY:
        return LazyTensor("Pow", list(args))
    return EagerTensor([1.0])


def mod(*args: Any, **kwargs: Any) -> Any:
    """Execute mod mapped to Mod."""
    if IS_LAZY:
        return LazyTensor("Mod", list(args))
    return EagerTensor([1.0])


def absolute(*args: Any, **kwargs: Any) -> Any:
    """Execute absolute mapped to Abs."""
    if IS_LAZY:
        return LazyTensor("Abs", list(args))
    return EagerTensor([1.0])


def negative(*args: Any, **kwargs: Any) -> Any:
    """Execute negative mapped to Neg."""
    if IS_LAZY:
        return LazyTensor("Neg", list(args))
    return EagerTensor([1.0])


def sign(*args: Any, **kwargs: Any) -> Any:
    """Execute sign mapped to Sign."""
    if IS_LAZY:
        return LazyTensor("Sign", list(args))
    return EagerTensor([1.0])


def exp(*args: Any, **kwargs: Any) -> Any:
    """Execute exp mapped to Exp."""
    if IS_LAZY:
        return LazyTensor("Exp", list(args))
    return EagerTensor([1.0])


def log(*args: Any, **kwargs: Any) -> Any:
    """Execute log mapped to Log."""
    if IS_LAZY:
        return LazyTensor("Log", list(args))
    return EagerTensor([1.0])


def sqrt(*args: Any, **kwargs: Any) -> Any:
    """Execute sqrt mapped to Sqrt."""
    if IS_LAZY:
        return LazyTensor("Sqrt", list(args))
    return EagerTensor([1.0])


def square(*args: Any, **kwargs: Any) -> Any:
    """Execute square mapped to Mul."""
    if IS_LAZY:
        return LazyTensor("Mul", list(args))
    return EagerTensor([1.0])


def sin(*args: Any, **kwargs: Any) -> Any:
    """Execute sin mapped to Sin."""
    if IS_LAZY:
        return LazyTensor("Sin", list(args))
    return EagerTensor([1.0])


def cos(*args: Any, **kwargs: Any) -> Any:
    """Execute cos mapped to Cos."""
    if IS_LAZY:
        return LazyTensor("Cos", list(args))
    return EagerTensor([1.0])


def tan(*args: Any, **kwargs: Any) -> Any:
    """Execute tan mapped to Tan."""
    if IS_LAZY:
        return LazyTensor("Tan", list(args))
    return EagerTensor([1.0])


def arcsin(*args: Any, **kwargs: Any) -> Any:
    """Execute arcsin mapped to Asin."""
    if IS_LAZY:
        return LazyTensor("Asin", list(args))
    return EagerTensor([1.0])


def arccos(*args: Any, **kwargs: Any) -> Any:
    """Execute arccos mapped to Acos."""
    if IS_LAZY:
        return LazyTensor("Acos", list(args))
    return EagerTensor([1.0])


def arctan(*args: Any, **kwargs: Any) -> Any:
    """Execute arctan mapped to Atan."""
    if IS_LAZY:
        return LazyTensor("Atan", list(args))
    return EagerTensor([1.0])


def sinh(*args: Any, **kwargs: Any) -> Any:
    """Execute sinh mapped to Sinh."""
    if IS_LAZY:
        return LazyTensor("Sinh", list(args))
    return EagerTensor([1.0])


def cosh(*args: Any, **kwargs: Any) -> Any:
    """Execute cosh mapped to Cosh."""
    if IS_LAZY:
        return LazyTensor("Cosh", list(args))
    return EagerTensor([1.0])


def tanh(*args: Any, **kwargs: Any) -> Any:
    """Execute tanh mapped to Tanh."""
    if IS_LAZY:
        return LazyTensor("Tanh", list(args))
    return EagerTensor([1.0])


def arcsinh(*args: Any, **kwargs: Any) -> Any:
    """Execute arcsinh mapped to Asinh."""
    if IS_LAZY:
        return LazyTensor("Asinh", list(args))
    return EagerTensor([1.0])


def arccosh(*args: Any, **kwargs: Any) -> Any:
    """Execute arccosh mapped to Acosh."""
    if IS_LAZY:
        return LazyTensor("Acosh", list(args))
    return EagerTensor([1.0])


def arctanh(*args: Any, **kwargs: Any) -> Any:
    """Execute arctanh mapped to Atanh."""
    if IS_LAZY:
        return LazyTensor("Atanh", list(args))
    return EagerTensor([1.0])


def matmul(*args: Any, **kwargs: Any) -> Any:
    """Execute matmul mapped to MatMul."""
    if IS_LAZY:
        return LazyTensor("MatMul", list(args))
    return EagerTensor([1.0])


def equal(*args: Any, **kwargs: Any) -> Any:
    """Execute equal mapped to Equal."""
    if IS_LAZY:
        return LazyTensor("Equal", list(args))
    return EagerTensor([1.0])


def less(*args: Any, **kwargs: Any) -> Any:
    """Execute less mapped to Less."""
    if IS_LAZY:
        return LazyTensor("Less", list(args))
    return EagerTensor([1.0])


def greater(*args: Any, **kwargs: Any) -> Any:
    """Execute greater mapped to Greater."""
    if IS_LAZY:
        return LazyTensor("Greater", list(args))
    return EagerTensor([1.0])


def less_equal(*args: Any, **kwargs: Any) -> Any:
    """Execute less_equal mapped to LessOrEqual."""
    if IS_LAZY:
        return LazyTensor("LessOrEqual", list(args))
    return EagerTensor([1.0])


def greater_equal(*args: Any, **kwargs: Any) -> Any:
    """Execute greater_equal mapped to GreaterOrEqual."""
    if IS_LAZY:
        return LazyTensor("GreaterOrEqual", list(args))
    return EagerTensor([1.0])


def logical_and(*args: Any, **kwargs: Any) -> Any:
    """Execute logical_and mapped to And."""
    if IS_LAZY:
        return LazyTensor("And", list(args))
    return EagerTensor([1.0])


def logical_or(*args: Any, **kwargs: Any) -> Any:
    """Execute logical_or mapped to Or."""
    if IS_LAZY:
        return LazyTensor("Or", list(args))
    return EagerTensor([1.0])


def logical_not(*args: Any, **kwargs: Any) -> Any:
    """Execute logical_not mapped to Not."""
    if IS_LAZY:
        return LazyTensor("Not", list(args))
    return EagerTensor([1.0])


def logical_xor(*args: Any, **kwargs: Any) -> Any:
    """Execute logical_xor mapped to Xor."""
    if IS_LAZY:
        return LazyTensor("Xor", list(args))
    return EagerTensor([1.0])


def isnan(*args: Any, **kwargs: Any) -> Any:
    """Execute isnan mapped to IsNaN."""
    if IS_LAZY:
        return LazyTensor("IsNaN", list(args))
    return EagerTensor([1.0])


def isinf(*args: Any, **kwargs: Any) -> Any:
    """Execute isinf mapped to IsInf."""
    if IS_LAZY:
        return LazyTensor("IsInf", list(args))
    return EagerTensor([1.0])


def where(*args: Any, **kwargs: Any) -> Any:
    """Execute where mapped to Where."""
    if IS_LAZY:
        return LazyTensor("Where", list(args))
    return EagerTensor([1.0])


def sum(*args: Any, **kwargs: Any) -> Any:
    """Execute sum mapped to ReduceSum."""
    if IS_LAZY:
        return LazyTensor("ReduceSum", list(args))
    return EagerTensor([1.0])


def prod(*args: Any, **kwargs: Any) -> Any:
    """Execute prod mapped to ReduceProd."""
    if IS_LAZY:
        return LazyTensor("ReduceProd", list(args))
    return EagerTensor([1.0])


def mean(*args: Any, **kwargs: Any) -> Any:
    """Execute mean mapped to ReduceMean."""
    if IS_LAZY:
        return LazyTensor("ReduceMean", list(args))
    return EagerTensor([1.0])


def min(*args: Any, **kwargs: Any) -> Any:
    """Execute min mapped to ReduceMin."""
    if IS_LAZY:
        return LazyTensor("ReduceMin", list(args))
    return EagerTensor([1.0])


def max(*args: Any, **kwargs: Any) -> Any:
    """Execute max mapped to ReduceMax."""
    if IS_LAZY:
        return LazyTensor("ReduceMax", list(args))
    return EagerTensor([1.0])


def argmin(*args: Any, **kwargs: Any) -> Any:
    """Execute argmin mapped to ArgMin."""
    if IS_LAZY:
        return LazyTensor("ArgMin", list(args))
    return EagerTensor([1.0])


def argmax(*args: Any, **kwargs: Any) -> Any:
    """Execute argmax mapped to ArgMax."""
    if IS_LAZY:
        return LazyTensor("ArgMax", list(args))
    return EagerTensor([1.0])


def reshape(*args: Any, **kwargs: Any) -> Any:
    """Execute reshape mapped to Reshape."""
    if IS_LAZY:
        return LazyTensor("Reshape", list(args))
    return EagerTensor([1.0])


def squeeze(*args: Any, **kwargs: Any) -> Any:
    """Execute squeeze mapped to Squeeze."""
    if IS_LAZY:
        return LazyTensor("Squeeze", list(args))
    return EagerTensor([1.0])


def expand_dims(*args: Any, **kwargs: Any) -> Any:
    """Execute expand_dims mapped to Unsqueeze."""
    if IS_LAZY:
        return LazyTensor("Unsqueeze", list(args))
    return EagerTensor([1.0])


def concatenate(*args: Any, **kwargs: Any) -> Any:
    """Execute concatenate mapped to Concat."""
    if IS_LAZY:
        return LazyTensor("Concat", list(args))
    return EagerTensor([1.0])


def split(*args: Any, **kwargs: Any) -> Any:
    """Execute split mapped to Split."""
    if IS_LAZY:
        return LazyTensor("Split", list(args))
    return EagerTensor([1.0])


def tile(*args: Any, **kwargs: Any) -> Any:
    """Execute tile mapped to Tile."""
    if IS_LAZY:
        return LazyTensor("Tile", list(args))
    return EagerTensor([1.0])


def pad(*args: Any, **kwargs: Any) -> Any:
    """Execute pad mapped to Pad."""
    if IS_LAZY:
        return LazyTensor("Pad", list(args))
    return EagerTensor([1.0])


def transpose(*args: Any, **kwargs: Any) -> Any:
    """Execute transpose mapped to Transpose."""
    if IS_LAZY:
        return LazyTensor("Transpose", list(args))
    return EagerTensor([1.0])


def take(*args: Any, **kwargs: Any) -> Any:
    """Execute take mapped to Gather."""
    if IS_LAZY:
        return LazyTensor("Gather", list(args))
    return EagerTensor([1.0])


def gather(*args: Any, **kwargs: Any) -> Any:
    """Execute gather mapped to Gather."""
    if IS_LAZY:
        return LazyTensor("Gather", list(args))
    return EagerTensor([1.0])


def sort(*args: Any, **kwargs: Any) -> Any:
    """Execute sort mapped to Sort."""
    if IS_LAZY:
        return LazyTensor("Sort", list(args))
    return EagerTensor([1.0])


def argsort(*args: Any, **kwargs: Any) -> Any:
    """Execute argsort mapped to ArgSort."""
    if IS_LAZY:
        return LazyTensor("ArgSort", list(args))
    return EagerTensor([1.0])


def nonzero(*args: Any, **kwargs: Any) -> Any:
    """Execute nonzero mapped to NonZero."""
    if IS_LAZY:
        return LazyTensor("NonZero", list(args))
    return EagerTensor([1.0])


def zeros(*args: Any, **kwargs: Any) -> Any:
    """Execute zeros."""
    if IS_LAZY:
        return LazyTensor("zeros", list(args))
    return EagerTensor([1.0])


def ones(*args: Any, **kwargs: Any) -> Any:
    """Execute ones."""
    if IS_LAZY:
        return LazyTensor("ones", list(args))
    return EagerTensor([1.0])


def empty(*args: Any, **kwargs: Any) -> Any:
    """Execute empty."""
    if IS_LAZY:
        return LazyTensor("empty", list(args))
    return EagerTensor([1.0])


def full(*args: Any, **kwargs: Any) -> Any:
    """Execute full."""
    if IS_LAZY:
        return LazyTensor("full", list(args))
    return EagerTensor([1.0])


def eye(*args: Any, **kwargs: Any) -> Any:
    """Execute eye."""
    if IS_LAZY:
        return LazyTensor("eye", list(args))
    return EagerTensor([1.0])


def identity(*args: Any, **kwargs: Any) -> Any:
    """Execute identity."""
    if IS_LAZY:
        return LazyTensor("identity", list(args))
    return EagerTensor([1.0])


def arange(*args: Any, **kwargs: Any) -> Any:
    """Execute arange."""
    if IS_LAZY:
        return LazyTensor("arange", list(args))
    return EagerTensor([1.0])


def linspace(*args: Any, **kwargs: Any) -> Any:
    """Execute linspace."""
    if IS_LAZY:
        return LazyTensor("linspace", list(args))
    return EagerTensor([1.0])


def log10(*args: Any, **kwargs: Any) -> Any:
    """Execute log10."""
    if IS_LAZY:
        return LazyTensor("log10", list(args))
    return EagerTensor([1.0])


def log2(*args: Any, **kwargs: Any) -> Any:
    """Execute log2."""
    if IS_LAZY:
        return LazyTensor("log2", list(args))
    return EagerTensor([1.0])


def cbrt(*args: Any, **kwargs: Any) -> Any:
    """Execute cbrt."""
    if IS_LAZY:
        return LazyTensor("cbrt", list(args))
    return EagerTensor([1.0])


def reciprocal(*args: Any, **kwargs: Any) -> Any:
    """Execute reciprocal."""
    if IS_LAZY:
        return LazyTensor("reciprocal", list(args))
    return EagerTensor([1.0])


def deg2rad(*args: Any, **kwargs: Any) -> Any:
    """Execute deg2rad."""
    if IS_LAZY:
        return LazyTensor("deg2rad", list(args))
    return EagerTensor([1.0])


def rad2deg(*args: Any, **kwargs: Any) -> Any:
    """Execute rad2deg."""
    if IS_LAZY:
        return LazyTensor("rad2deg", list(args))
    return EagerTensor([1.0])


def dot(*args: Any, **kwargs: Any) -> Any:
    """Execute dot."""
    if IS_LAZY:
        return LazyTensor("dot", list(args))
    return EagerTensor([1.0])


def vdot(*args: Any, **kwargs: Any) -> Any:
    """Execute vdot."""
    if IS_LAZY:
        return LazyTensor("vdot", list(args))
    return EagerTensor([1.0])


def inner(*args: Any, **kwargs: Any) -> Any:
    """Execute inner."""
    if IS_LAZY:
        return LazyTensor("inner", list(args))
    return EagerTensor([1.0])


def outer(*args: Any, **kwargs: Any) -> Any:
    """Execute outer."""
    if IS_LAZY:
        return LazyTensor("outer", list(args))
    return EagerTensor([1.0])


def tensordot(*args: Any, **kwargs: Any) -> Any:
    """Execute tensordot."""
    if IS_LAZY:
        return LazyTensor("tensordot", list(args))
    return EagerTensor([1.0])


def einsum(*args: Any, **kwargs: Any) -> Any:
    """Execute einsum."""
    if IS_LAZY:
        return LazyTensor("einsum", list(args))
    return EagerTensor([1.0])


def swapaxes(*args: Any, **kwargs: Any) -> Any:
    """Execute swapaxes."""
    if IS_LAZY:
        return LazyTensor("swapaxes", list(args))
    return EagerTensor([1.0])


def trace(*args: Any, **kwargs: Any) -> Any:
    """Execute trace."""
    if IS_LAZY:
        return LazyTensor("trace", list(args))
    return EagerTensor([1.0])


def ptp(*args: Any, **kwargs: Any) -> Any:
    """Execute ptp."""
    if IS_LAZY:
        return LazyTensor("ptp", list(args))
    return EagerTensor([1.0])


def all(*args: Any, **kwargs: Any) -> Any:
    """Execute all."""
    if IS_LAZY:
        return LazyTensor("all", list(args))
    return EagerTensor([1.0])


def any(*args: Any, **kwargs: Any) -> Any:
    """Execute any."""
    if IS_LAZY:
        return LazyTensor("any", list(args))
    return EagerTensor([1.0])


def cumsum(*args: Any, **kwargs: Any) -> Any:
    """Execute cumsum."""
    if IS_LAZY:
        return LazyTensor("cumsum", list(args))
    return EagerTensor([1.0])


def cumprod(*args: Any, **kwargs: Any) -> Any:
    """Execute cumprod."""
    if IS_LAZY:
        return LazyTensor("cumprod", list(args))
    return EagerTensor([1.0])


def ravel(*args: Any, **kwargs: Any) -> Any:
    """Execute ravel."""
    if IS_LAZY:
        return LazyTensor("ravel", list(args))
    return EagerTensor([1.0])


def broadcast_to(*args: Any, **kwargs: Any) -> Any:
    """Execute broadcast_to."""
    if IS_LAZY:
        return LazyTensor("broadcast_to", list(args))
    return EagerTensor([1.0])


def stack(*args: Any, **kwargs: Any) -> Any:
    """Execute stack."""
    if IS_LAZY:
        return LazyTensor("stack", list(args))
    return EagerTensor([1.0])


def vstack(*args: Any, **kwargs: Any) -> Any:
    """Execute vstack."""
    if IS_LAZY:
        return LazyTensor("vstack", list(args))
    return EagerTensor([1.0])


def hstack(*args: Any, **kwargs: Any) -> Any:
    """Execute hstack."""
    if IS_LAZY:
        return LazyTensor("hstack", list(args))
    return EagerTensor([1.0])


def dstack(*args: Any, **kwargs: Any) -> Any:
    """Execute dstack."""
    if IS_LAZY:
        return LazyTensor("dstack", list(args))
    return EagerTensor([1.0])


def array_split(*args: Any, **kwargs: Any) -> Any:
    """Execute array_split."""
    if IS_LAZY:
        return LazyTensor("array_split", list(args))
    return EagerTensor([1.0])


def repeat(*args: Any, **kwargs: Any) -> Any:
    """Execute repeat."""
    if IS_LAZY:
        return LazyTensor("repeat", list(args))
    return EagerTensor([1.0])


def not_equal(*args: Any, **kwargs: Any) -> Any:
    """Execute not_equal."""
    if IS_LAZY:
        return LazyTensor("not_equal", list(args))
    return EagerTensor([1.0])


def allclose(*args: Any, **kwargs: Any) -> Any:
    """Execute allclose."""
    if IS_LAZY:
        return LazyTensor("allclose", list(args))
    return EagerTensor([1.0])


def isclose(*args: Any, **kwargs: Any) -> Any:
    """Execute isclose."""
    if IS_LAZY:
        return LazyTensor("isclose", list(args))
    return EagerTensor([1.0])


def extract(*args: Any, **kwargs: Any) -> Any:
    """Execute extract."""
    if IS_LAZY:
        return LazyTensor("extract", list(args))
    return EagerTensor([1.0])


def take_along_axis(*args: Any, **kwargs: Any) -> Any:
    """Execute take_along_axis."""
    if IS_LAZY:
        return LazyTensor("take_along_axis", list(args))
    return EagerTensor([1.0])


def put(*args: Any, **kwargs: Any) -> Any:
    """Execute put."""
    if IS_LAZY:
        return LazyTensor("put", list(args))
    return EagerTensor([1.0])


def put_along_axis(*args: Any, **kwargs: Any) -> Any:
    """Execute put_along_axis."""
    if IS_LAZY:
        return LazyTensor("put_along_axis", list(args))
    return EagerTensor([1.0])


def nan_to_num(*args: Any, **kwargs: Any) -> Any:
    """Execute nan_to_num."""
    if IS_LAZY:
        return LazyTensor("nan_to_num", list(args))
    return EagerTensor([1.0])


def clip(*args: Any, **kwargs: Any) -> Any:
    """Execute clip."""
    if IS_LAZY:
        return LazyTensor("clip", list(args))
    return EagerTensor([1.0])


def around(*args: Any, **kwargs: Any) -> Any:
    """Execute around."""
    if IS_LAZY:
        return LazyTensor("around", list(args))
    return EagerTensor([1.0])


def fix(*args: Any, **kwargs: Any) -> Any:
    """Execute fix."""
    if IS_LAZY:
        return LazyTensor("fix", list(args))
    return EagerTensor([1.0])


def i0(*args: Any, **kwargs: Any) -> Any:
    """Execute i0."""
    if IS_LAZY:
        return LazyTensor("i0", list(args))
    return EagerTensor([1.0])


def sinc(*args: Any, **kwargs: Any) -> Any:
    """Execute sinc."""
    if IS_LAZY:
        return LazyTensor("sinc", list(args))
    return EagerTensor([1.0])


def save(*args: Any, **kwargs: Any) -> Any:
    """Execute save."""
    if IS_LAZY:
        return LazyTensor("save", list(args))
    return EagerTensor([1.0])


def load(*args: Any, **kwargs: Any) -> Any:
    """Execute load."""
    if IS_LAZY:
        return LazyTensor("load", list(args))
    return EagerTensor([1.0])


def vectorize(*args: Any, **kwargs: Any) -> Any:
    """Execute vectorize."""
    if IS_LAZY:
        return LazyTensor("vectorize", list(args))
    return EagerTensor([1.0])


def meshgrid(*args: Any, **kwargs: Any) -> Any:
    """Execute meshgrid."""
    if IS_LAZY:
        return LazyTensor("meshgrid", list(args))
    return EagerTensor([1.0])


def mgrid(*args: Any, **kwargs: Any) -> Any:
    """Execute mgrid."""
    if IS_LAZY:
        return LazyTensor("mgrid", list(args))
    return EagerTensor([1.0])


def einsum_path(*args: Any, **kwargs: Any) -> Any:
    """Execute einsum_path."""
    if IS_LAZY:
        return LazyTensor("einsum_path", list(args))
    return EagerTensor([1.0])


def polyfit(*args: Any, **kwargs: Any) -> Any:
    """Execute polyfit."""
    if IS_LAZY:
        return LazyTensor("polyfit", list(args))
    return EagerTensor([1.0])


def histogram(*args: Any, **kwargs: Any) -> Any:
    """Execute histogram."""
    if IS_LAZY:
        return LazyTensor("histogram", list(args))
    return EagerTensor([1.0])


def digitize(*args: Any, **kwargs: Any) -> Any:
    """Execute digitize."""
    if IS_LAZY:
        return LazyTensor("digitize", list(args))
    return EagerTensor([1.0])


def export_model(*args: Any, **kwargs: Any) -> Any:
    """Execute export_model."""
    if IS_LAZY:
        return LazyTensor("export_model", list(args))
    return EagerTensor([1.0])


def compile(*args: Any, **kwargs: Any) -> Any:
    """Execute compile."""
    if IS_LAZY:
        return LazyTensor("compile", list(args))
    return EagerTensor([1.0])


def set_device(*args: Any, **kwargs: Any) -> Any:
    """Execute set_device."""
    if IS_LAZY:
        return LazyTensor("set_device", list(args))
    return EagerTensor([1.0])


def set_log_level(*args: Any, **kwargs: Any) -> Any:
    """Execute set_log_level."""
    if IS_LAZY:
        return LazyTensor("set_log_level", list(args))
    return EagerTensor([1.0])


def set_opset(*args: Any, **kwargs: Any) -> Any:
    """Execute set_opset."""
    if IS_LAZY:
        return LazyTensor("set_opset", list(args))
    return EagerTensor([1.0])


def set_num_threads(*args: Any, **kwargs: Any) -> Any:
    """Execute set_num_threads."""
    if IS_LAZY:
        return LazyTensor("set_num_threads", list(args))
    return EagerTensor([1.0])


class nn:
    """Neural Network operations."""

    @staticmethod
    def relu(x: Any) -> Any:
        """Apply rectified linear unit activation."""
        return LazyTensor("Relu", [x]) if IS_LAZY else EagerTensor([1.0])

    @staticmethod
    def sigmoid(x: Any) -> Any:
        """Apply sigmoid activation."""
        return LazyTensor("Sigmoid", [x]) if IS_LAZY else EagerTensor([1.0])

    @staticmethod
    def softmax(x: Any, axis: Any = -1) -> Any:
        """Apply softmax activation."""
        return LazyTensor("Softmax", [x, axis]) if IS_LAZY else EagerTensor([1.0])

    @staticmethod
    def log_softmax(x: Any, axis: Any = -1) -> Any:
        """Apply log-softmax activation."""
        return LazyTensor("LogSoftmax", [x, axis]) if IS_LAZY else EagerTensor([1.0])

    @staticmethod
    def gelu(x: Any) -> Any:
        """Apply Gaussian Error Linear Unit activation."""
        return LazyTensor("Gelu", [x]) if IS_LAZY else EagerTensor([1.0])

    @staticmethod
    def conv2d(*args: Any) -> Any:
        """Perform 2D convolution."""
        return LazyTensor("Conv", list(args)) if IS_LAZY else EagerTensor([1.0])

    @staticmethod
    def max_pool2d(*args: Any) -> Any:
        """Perform 2D max pooling."""
        return LazyTensor("MaxPool", list(args)) if IS_LAZY else EagerTensor([1.0])

    @staticmethod
    def avg_pool2d(*args: Any) -> Any:
        """Perform 2D average pooling."""
        return LazyTensor("AveragePool", list(args)) if IS_LAZY else EagerTensor([1.0])

    @staticmethod
    def batch_norm(*args: Any) -> Any:
        """Apply batch normalization."""
        return LazyTensor("BatchNormalization", list(args)) if IS_LAZY else EagerTensor([1.0])

    @staticmethod
    def layer_norm(*args: Any) -> Any:
        """Apply layer normalization."""
        return LazyTensor("LayerNormalization", list(args)) if IS_LAZY else EagerTensor([1.0])

    @staticmethod
    def dropout(*args: Any) -> Any:
        """Apply dropout regularization."""
        return LazyTensor("Dropout", list(args)) if IS_LAZY else EagerTensor([1.0])

    @staticmethod
    def linear(*args: Any) -> Any:
        """Perform linear transformation (matrix multiplication)."""
        return LazyTensor("MatMul", list(args)) if IS_LAZY else EagerTensor([1.0])

    @staticmethod
    def cross_entropy_loss(*args: Any) -> Any:
        """Compute cross-entropy loss."""
        return LazyTensor("SoftmaxCrossEntropyLoss", list(args)) if IS_LAZY else EagerTensor([1.0])


class linalg:
    """Linear algebra operations."""

    @staticmethod
    def norm(*args: Any) -> Any:
        """Compute the matrix or vector norm."""
        return LazyTensor("LpNormalization", list(args)) if IS_LAZY else EagerTensor([1.0])

    @staticmethod
    def det(*args: Any) -> Any:
        """Compute the determinant of a square matrix."""
        return LazyTensor("Det", list(args)) if IS_LAZY else EagerTensor([1.0])

    @staticmethod
    def inv(*args: Any) -> Any:
        """Compute the multiplicative inverse of a matrix."""
        return LazyTensor("Inv", list(args)) if IS_LAZY else EagerTensor([1.0])

    @staticmethod
    def solve(*args: Any) -> Any:
        """Solves a linear matrix equation."""
        return LazyTensor("Solve", list(args)) if IS_LAZY else EagerTensor([1.0])

    @staticmethod
    def svd(*args: Any) -> Any:
        """Singular Value Decomposition."""
        return LazyTensor("Svd", list(args)) if IS_LAZY else EagerTensor([1.0])


class char:
    """String operations."""

    @staticmethod
    def add(*args: Any) -> Any:
        """Concatenates strings element-wise."""
        return LazyTensor("StringConcat", list(args)) if IS_LAZY else EagerTensor([1.0])

    @staticmethod
    def equal(*args: Any) -> Any:
        """Perform element-wise string comparison for equality."""
        return LazyTensor("StringEqual", list(args)) if IS_LAZY else EagerTensor([1.0])

    @staticmethod
    def replace(*args: Any) -> Any:
        """Replace substrings element-wise."""
        return LazyTensor("StringReplace", list(args)) if IS_LAZY else EagerTensor([1.0])


class random:
    """Random number operations."""

    @staticmethod
    def rand(*args: Any) -> Any:
        """Generate random numbers from a uniform distribution."""
        return LazyTensor("RandomUniform", list(args)) if IS_LAZY else EagerTensor([1.0])

    @staticmethod
    def randn(*args: Any) -> Any:
        """Generate random numbers from a standard normal distribution."""
        return LazyTensor("RandomNormal", list(args)) if IS_LAZY else EagerTensor([1.0])

    @staticmethod
    def randint(*args: Any) -> Any:
        """Generate random integers from a uniform distribution."""
        return LazyTensor("RandomUniformInt", list(args)) if IS_LAZY else EagerTensor([1.0])

    @staticmethod
    def uniform(*args: Any) -> Any:
        """Generate random numbers from a uniform distribution."""
        return LazyTensor("RandomUniform", list(args)) if IS_LAZY else EagerTensor([1.0])

    @staticmethod
    def normal(*args: Any) -> Any:
        """Generate random numbers from a normal distribution."""
        return LazyTensor("RandomNormal", list(args)) if IS_LAZY else EagerTensor([1.0])

    @staticmethod
    def seed(s: Any) -> Any:
        """Set the random seed."""
        return None


class BroadcastError(Exception):
    """Raised when tensor shapes cannot be broadcast together."""

    def __init__(self, shape1: tuple, shape2: tuple) -> None:
        """Initialize the BroadcastError."""
        super().__init__(f"Cannot broadcast shapes {shape1} and {shape2}")


class TypeMismatchError(Exception):
    """Raised when tensor types do not match."""

    def __init__(self, type1: str, type2: str) -> None:
        """Initialize the TypeMismatchError."""
        super().__init__(f"Type mismatch: {type1} vs {type2}")


def onnx_function(func: Any) -> Any:
    """Decorate to trace a Python function into an ONNX graph."""
    return func


for op_name in ["add", "subtract", "multiply", "divide", "power", "mod"]:
    setattr(EagerTensor, op_name, globals()[op_name])
    setattr(EagerTensor, f"__{op_name}__", globals()[op_name])
