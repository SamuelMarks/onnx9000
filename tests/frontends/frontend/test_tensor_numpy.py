"""Module providing core logic and structural definitions."""

import numpy as np
from onnx9000.frontends.frontend.tensor import Tensor
from onnx9000.core.dtypes import DType
from onnx9000.frontends.frontend.tracer import trace


def test_numpy_dispatch():
    """Provides semantic functionality and verification."""

    def my_np_func(x):
        """Provides semantic functionality and verification."""
        return np.add(x, np.array([2.0], dtype=np.float32))

    x = Tensor((2,), DType.FLOAT32, "x")
    builder = trace(my_np_func, x)
    assert len(builder.nodes) == 1
    assert builder.nodes[0].op_type == "Add"


def test_dynamic_axes_trace():
    """Provides semantic functionality and verification."""
    from onnx9000.frontends.frontend.tracer import trace

    def my_func(x):
        """Provides semantic functionality and verification."""
        return x * 2.0

    x = Tensor(("N", 3, 224, 224), DType.FLOAT32, "x")
    builder = trace(my_func, x)
    assert builder.inputs[0].shape[0] == "N"
