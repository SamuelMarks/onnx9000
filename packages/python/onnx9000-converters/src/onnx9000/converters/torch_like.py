"""PyTorch-like drop-in replacement namespace."""

from onnx9000.converters.frontend import nn
from onnx9000.converters.frontend.exporter import export
from onnx9000.converters.frontend.tensor import Parameter, Tensor
from onnx9000.converters.frontend.tracer import script, trace
from onnx9000.core.dtypes import DType


def tensor(data, dtype=None):
    """Implement the tensor method or operation."""
    if dtype is None:
        return Tensor(data=data)
    return Tensor(data=data, dtype=dtype)


def zeros(*shape, dtype=DType.FLOAT32):
    """Implement the zeros method or operation."""
    import numpy as np

    return Tensor(shape=shape, dtype=dtype, data=np.zeros(shape, dtype=np.float32))


def ones(*shape, dtype=DType.FLOAT32):
    """Implement the ones method or operation."""
    import numpy as np

    return Tensor(shape=shape, dtype=dtype, data=np.ones(shape, dtype=np.float32))


def randn(*shape, dtype=DType.FLOAT32):
    """Implement the randn method or operation."""
    import numpy as np

    return Tensor(shape=shape, dtype=dtype, data=np.random.randn(*shape).astype(np.float32))


float32 = DType.FLOAT32
float64 = DType.FLOAT64
int32 = DType.INT32
int64 = DType.INT64
bool = DType.BOOL


class jit:
    """Class jit implementation."""

    @staticmethod
    def trace(*args, **kwargs):
        """Implement the trace method or operation."""
        if len(args) == 2:
            return trace(args[0], args[1])
        return trace(*args, **kwargs)

    @staticmethod
    def script(*args, **kwargs):
        """Implement the script method or operation."""
        return script(*args, **kwargs)


class onnx:
    """Class onnx implementation."""

    @staticmethod
    def export(*args, **kwargs):
        """Implement the export method or operation."""
        return export(*args, **kwargs)


__all__ = [
    "Tensor",
    "Parameter",
    "nn",
    "export",
    "trace",
    "script",
    "tensor",
    "zeros",
    "ones",
    "randn",
    "float32",
    "float64",
    "int32",
    "int64",
    "bool",
    "jit",
    "onnx",
]
