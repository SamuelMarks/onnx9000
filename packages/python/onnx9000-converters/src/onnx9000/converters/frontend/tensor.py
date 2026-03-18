"""
Frontend Sub-Package
Provides tracing and PyTorch-like interfaces to define and capture
computation graphs from native Python execution.
"""

from typing import Any, Optional, Union

import numpy as np
from onnx9000.core.dtypes import DType


class Node:
    """Class Node implementation."""

    def __init__(
        self,
        op_type: str,
        inputs: list[Any],
        outputs: list[Any],
        attributes: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        domain: str = "",
    ) -> None:
        """Initializes the frontend builder or trace context."""
        self.op_type = op_type
        self.domain = domain
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes or {}
        self.name = name or ""


class Tensor:
    """Symbolic representation of data in the frontend graph."""

    _id_counter = 0

    def __init__(
        self,
        shape: Optional[tuple[Union[int, str]]] = None,
        dtype: Optional[DType] = None,
        name: Optional[str] = None,
        domain: str = "",
        data: Optional[Any] = None,
    ) -> None:
        """Initializes the frontend builder or trace context."""
        if data is not None:
            if not isinstance(data, np.ndarray):
                try:
                    data = np.asarray(data)
                except Exception:
                    data = None
            if isinstance(data, np.ndarray):
                if shape is None:
                    shape = data.shape
                if dtype is None:
                    if data.dtype == np.int64:
                        dtype = DType.INT64
                    elif data.dtype == np.int32:
                        dtype = DType.INT32
                    elif data.dtype == np.float64:
                        dtype = DType.FLOAT64
                    elif data.dtype == bool:
                        dtype = DType.BOOL
                    else:
                        dtype = DType.FLOAT32
        self._shape = shape if shape is not None else ()
        self._dtype = dtype if dtype is not None else DType.FLOAT32
        self.data = data
        if name is None:
            Tensor._id_counter += 1
            self._name = f"tensor_{Tensor._id_counter}"
        else:
            self._name = name

    @property
    def shape(self) -> tuple[Union[int, str]]:
        """Returns the shape of the tensor as a tuple of integers."""
        return self._shape

    @property
    def dtype(self) -> DType:
        """Returns the data type (DType) of the tensor."""
        return self._dtype

    @property
    def name(self) -> str:
        """Returns the name of the tensor. Typically generated during tracing."""
        return self._name

    @property
    def T(self) -> "Tensor":
        """Returns a transposed version of the tensor (swapping the last two dimensions if 2D+)."""
        return self._op("Transpose")

    def __repr__(self) -> str:
        """Returns a string representation of the tensor, including its name, shape, and dtype."""
        return f"Tensor(name={self.name}, shape={self.shape}, dtype={self.dtype})"

    def _op(self, op_type: str, *args, **kwargs) -> Any:
        """A helper method to apply a unary or binary operation to the tensor, generating a new tensor."""
        from onnx9000.converters.frontend.utils import record_op

        inputs = [self] + list(args)
        return record_op(op_type, inputs, attributes=kwargs)

    def __add__(self, other: Any) -> "Tensor":
        """Performs element-wise addition with another tensor or scalar."""
        return self._op("Add", other)

    def __radd__(self, other: Any) -> "Tensor":
        """Performs element-wise addition with another tensor or scalar (right-sided)."""
        return self._op("Add", other)

    def __sub__(self, other: Any) -> "Tensor":
        """Performs element-wise subtraction with another tensor or scalar."""
        return self._op("Sub", other)

    def __rsub__(self, other: Any) -> "Tensor":
        """Performs element-wise subtraction with another tensor or scalar (right-sided)."""
        from onnx9000.converters.frontend.utils import record_op

        return record_op("Sub", [other, self])

    def __mul__(self, other: Any) -> "Tensor":
        """Performs element-wise multiplication with another tensor or scalar."""
        return self._op("Mul", other)

    def __rmul__(self, other: Any) -> "Tensor":
        """Performs element-wise multiplication with another tensor or scalar (right-sided)."""
        return self._op("Mul", other)

    def __truediv__(self, other: Any) -> "Tensor":
        """Performs element-wise true division with another tensor or scalar."""
        return self._op("Div", other)

    def __rtruediv__(self, other: Any) -> "Tensor":
        """Performs element-wise true division with another tensor or scalar (right-sided)."""
        from onnx9000.converters.frontend.utils import record_op

        return record_op("Div", [other, self])

    def __matmul__(self, other: Any) -> "Tensor":
        """Performs matrix multiplication with another tensor."""
        return self._op("MatMul", other)

    def __pow__(self, other: Any) -> "Tensor":
        """Raises the tensor to the power of another tensor or scalar."""
        return self._op("Pow", other)

    def __mod__(self, other: Any) -> "Tensor":
        """Performs element-wise modulo with another tensor or scalar."""
        return self._op("Mod", other)

    def __neg__(self) -> "Tensor":
        """Returns the element-wise negation of the tensor."""
        return self._op("Neg")

    def __abs__(self) -> "Tensor":
        """Returns the element-wise absolute value of the tensor."""
        return self._op("Abs")

    def __getitem__(self, idx: Any) -> "Tensor":
        """Returns a slice or element of the tensor based on the provided index."""
        return self._op("Gather", idx)

    def __setitem__(self, idx: Any, value: Any) -> None:
        """Implements the __setitem__ operation for the tensor."""
        self._op("ScatterND", value, indices=idx)

    def __bool__(self) -> bool:
        """Implements the __bool__ operation for the tensor."""
        raise RuntimeError("Data-dependent control flow is not supported")

    def __and__(self, other: Any) -> "Tensor":
        """Implements the __and__ operation for the tensor."""
        return self._op("And", other)

    def __or__(self, other: Any) -> "Tensor":
        """Implements the __or__ operation for the tensor."""
        return self._op("Or", other)

    def __xor__(self, other: Any) -> "Tensor":
        """Implements the __xor__ operation for the tensor."""
        return self._op("Xor", other)

    def __invert__(self) -> "Tensor":
        """Implements the __invert__ operation for the tensor."""
        return self._op("Not")

    def __eq__(self, other: Any) -> "Tensor":
        """Returns a boolean tensor containing element-wise equality."""
        return self._op("Equal", other)

    def __ne__(self, other: Any) -> "Tensor":
        """Returns a boolean tensor containing element-wise inequality."""
        return self._op("Not", self._op("Equal", other))

    def __lt__(self, other: Any) -> "Tensor":
        """Returns a boolean tensor indicating if elements are strictly less than another."""
        return self._op("Less", other)

    def __le__(self, other: Any) -> "Tensor":
        """Returns a boolean tensor indicating if elements are less than or equal to another."""
        return self._op("LessOrEqual", other)

    def __gt__(self, other: Any) -> "Tensor":
        """Returns a boolean tensor indicating if elements are strictly greater than another."""
        return self._op("Greater", other)

    def __ge__(self, other: Any) -> "Tensor":
        """Returns a boolean tensor indicating if elements are greater than or equal to another."""
        return self._op("GreaterOrEqual", other)

    def sum(self, dim=None, keepdim=False) -> "Tensor":
        """Returns the sum of all elements in the tensor, or along a specified dimension."""
        return self._op("ReduceSum", axes=dim, keepdims=keepdim)

    def mean(self, dim=None, keepdim=False) -> "Tensor":
        """Returns the mean of all elements in the tensor, or along a specified dimension."""
        return self._op("ReduceMean", axes=dim, keepdims=keepdim)

    def max(self, dim=None, keepdim=False) -> "Tensor":
        """Returns the maximum value of all elements in the tensor, or along a specified dimension."""
        return self._op("ReduceMax", axes=dim, keepdims=keepdim)

    def min(self, dim=None, keepdim=False) -> "Tensor":
        """Returns the minimum value of all elements in the tensor, or along a specified dimension."""
        return self._op("ReduceMin", axes=dim, keepdims=keepdim)

    def transpose(self, dim0, dim1) -> "Tensor":
        """Transposes the tensor by swapping two specified dimensions."""
        ndim = len(self.shape)
        if ndim == 0:
            perm = []
        else:
            dim0 = dim0 if dim0 >= 0 else ndim + dim0
            dim1 = dim1 if dim1 >= 0 else ndim + dim1
            perm = list(range(ndim))
            (perm[dim0], perm[dim1]) = (perm[dim1], perm[dim0])
        return self._op("Transpose", perm=perm)

    def reshape(self, shape) -> "Tensor":
        """Returns a new tensor with the same data but a different shape."""
        return self._op("Reshape", shape)

    def view(self, *shape) -> "Tensor":
        """Returns a new view of the tensor with a different shape. Semantically similar to reshape."""
        return self.reshape(shape)

    def squeeze(self, dim=None) -> "Tensor":
        """Returns a tensor with all dimensions of size 1 removed, or a specific dimension removed."""
        return self._op("Squeeze", axes=dim)

    def unsqueeze(self, dim) -> "Tensor":
        """Returns a new tensor with a dimension of size one inserted at the specified position."""
        return self._op("Unsqueeze", axes=dim)

    def flatten(self, start_dim=0, end_dim=-1) -> "Tensor":
        """Flattens the tensor into a 1D tensor."""
        return self._op("Flatten", axis=start_dim)

    def expand(self, *sizes) -> "Tensor":
        """Implements the expand operation for the tensor."""
        return self._op("Expand", sizes)

    def broadcast_to(self, shape) -> "Tensor":
        """Implements the broadcast_to operation for the tensor."""
        return self.expand(*shape)

    def contiguous(self) -> "Tensor":
        """Returns a contiguous in memory tensor containing the same data."""
        return self

    def type(self, dtype) -> "Tensor":
        """Implements the type operation for the tensor."""
        return self._op("Cast", to=dtype)

    def to(self, *args, **kwargs) -> "Tensor":
        """Performs Tensor dtype and/or device conversion."""
        return self

    def exp(self) -> "Tensor":
        """Returns a new tensor with the exponential of the elements."""
        return self._op("Exp")

    def log(self) -> "Tensor":
        """Returns a new tensor with the natural logarithm of the elements."""
        return self._op("Log")

    def sqrt(self) -> "Tensor":
        """Implements the sqrt operation for the tensor."""
        return self._op("Sqrt")

    def sin(self) -> "Tensor":
        """Returns a new tensor with the sine of the elements."""
        return self._op("Sin")

    def cos(self) -> "Tensor":
        """Returns a new tensor with the cosine of the elements."""
        return self._op("Cos")

    def tan(self) -> "Tensor":
        """Returns a new tensor with the tangent of the elements."""
        return self._op("Tan")

    def asin(self) -> "Tensor":
        """Implements the asin operation for the tensor."""
        return self._op("Asin")

    def acos(self) -> "Tensor":
        """Implements the acos operation for the tensor."""
        return self._op("Acos")

    def atan(self) -> "Tensor":
        """Implements the atan operation for the tensor."""
        return self._op("Atan")

    def sinh(self) -> "Tensor":
        """Implements the sinh operation for the tensor."""
        return self._op("Sinh")

    def cosh(self) -> "Tensor":
        """Implements the cosh operation for the tensor."""
        return self._op("Cosh")

    def relu(self) -> "Tensor":
        """Returns a new tensor with the rectified linear unit (ReLU) applied to the elements."""
        return self._op("Relu")

    def sigmoid(self) -> "Tensor":
        """Returns a new tensor with the sigmoid function applied to the elements."""
        return self._op("Sigmoid")

    def tanh(self) -> "Tensor":
        """Returns a new tensor with the hyperbolic tangent of the elements."""
        return self._op("Tanh")

    def gelu(self) -> "Tensor":
        """Returns a new tensor with the Gaussian Error Linear Unit (GELU) applied to the elements."""
        return self._op("Gelu")

    def softmax(self, dim=None) -> "Tensor":
        """Applies the softmax function to an n-dimensional input tensor along a specified dimension."""
        return self._op("Softmax", axis=dim)

    def where(self, condition, y) -> "Tensor":
        """Implements the where operation for the tensor."""
        return self._op("Where", condition, y)

    def clip(self, min=None, max=None, min_val=None, max_val=None) -> "Tensor":
        """Implements the clip operation for the tensor."""
        return self._op(
            "Clip", min if min is not None else min_val, max if max is not None else max_val
        )

    def clamp(self, min=None, max=None, min_val=None, max_val=None) -> "Tensor":
        """Implements the clamp operation for the tensor."""
        return self.clip(min if min is not None else min_val, max if max is not None else max_val)

    def argmax(self, dim=None, keepdim=False) -> "Tensor":
        """Returns the indices of the maximum values along an axis."""
        return self._op("ArgMax", axis=dim, keepdims=keepdim)

    def argmin(self, dim=None, keepdim=False) -> "Tensor":
        """Returns the indices of the minimum values along an axis."""
        return self._op("ArgMin", axis=dim, keepdims=keepdim)

    def gather(self, dim, index) -> "Tensor":
        """Implements the gather operation for the tensor."""
        return self._op("Gather", index, axis=dim)

    def scatter(self, dim, index, src) -> "Tensor":
        """Implements the scatter operation for the tensor."""
        return self._op("Scatter", index, src, axis=dim)

    def masked_select(self, mask) -> "Tensor":
        """Implements the masked_select operation for the tensor."""
        return self._op("MaskedSelect", mask)

    def nonzero(self) -> "Tensor":
        """Implements the nonzero operation for the tensor."""
        return self._op("NonZero")

    def item(self) -> Any:
        """Returns the value of this tensor as a standard Python number. This only works for tensors with one element."""
        return self.data.item() if self.data is not None else None

    def tolist(self) -> list:
        """Returns the tensor as a (nested) list."""
        return self.data.tolist() if self.data is not None else None

    def numpy(self) -> np.ndarray:
        """Returns the tensor as a NumPy array."""
        return self.data if self.data is not None else None

    def requires_grad_(self, requires_grad=True) -> "Tensor":
        """Implements the requires_grad operation for the tensor."""
        return self

    @property
    def requires_grad(self) -> bool:
        """Implements the requires_grad operation for the tensor."""
        return getattr(self, "_requires_grad", False)

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """Implements the requires_grad operation for the tensor."""
        self._requires_grad = value

    @property
    def grad(self) -> Optional["Tensor"]:
        """Implements the grad operation for the tensor."""
        return getattr(self, "_grad", None)

    @grad.setter
    def grad(self, value: Optional["Tensor"]) -> None:
        """Implements the grad operation for the tensor."""
        self._grad = value

    def detach(self) -> "Tensor":
        """Returns a new Tensor, detached from the current graph."""
        return self

    def clone(self) -> "Tensor":
        """Returns a copy of the tensor with a new memory allocation."""
        return self


class Parameter(Tensor):
    """Subclass of Tensor denoting trainable weights or fixed initializers."""
