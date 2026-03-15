"""PyTorch-like drop-in replacement namespace."""

from onnx9000.frontends.frontend.tensor import Tensor, Parameter
from onnx9000.core.dtypes import DType
from onnx9000.frontends.frontend import nn
from onnx9000.frontends.frontend.exporter import export
from onnx9000.frontends.frontend.tracer import trace, script


def tensor(data, dtype=None):
    """Provides semantic functionality and verification."""
    if dtype is None:
        return Tensor(data=data)
    return Tensor(data=data, dtype=dtype)


def zeros(*shape, dtype=DType.FLOAT32):
    """Provides semantic functionality and verification."""
    import numpy as np

    return Tensor(shape=shape, dtype=dtype, data=np.zeros(shape, dtype=np.float32))


def ones(*shape, dtype=DType.FLOAT32):
    """Provides semantic functionality and verification."""
    import numpy as np

    return Tensor(shape=shape, dtype=dtype, data=np.ones(shape, dtype=np.float32))


def randn(*shape, dtype=DType.FLOAT32):
    """Provides semantic functionality and verification."""
    import numpy as np

    return Tensor(
        shape=shape, dtype=dtype, data=np.random.randn(*shape).astype(np.float32)
    )


float32 = DType.FLOAT32
float64 = DType.FLOAT64
int32 = DType.INT32
int64 = DType.INT64
bool = DType.BOOL


class jit:
    """Provides semantic functionality and verification."""

    @staticmethod
    def trace(*args, **kwargs):
        """Provides semantic functionality and verification."""
        if len(args) == 2:
            return trace(args[0], args[1])
        return trace(*args, **kwargs)

    @staticmethod
    def script(*args, **kwargs):
        """Provides semantic functionality and verification."""
        return script(*args, **kwargs)


class onnx:
    """Provides semantic functionality and verification."""

    @staticmethod
    def export(*args, **kwargs):
        """Provides semantic functionality and verification."""
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
