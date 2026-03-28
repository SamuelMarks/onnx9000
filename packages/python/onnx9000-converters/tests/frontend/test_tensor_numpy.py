"""Module providing core logic and structural definitions."""

import numpy as np
from onnx9000.converters.frontend.tensor import Tensor
from onnx9000.converters.frontend.tracer import trace
from onnx9000.core.dtypes import DType


def test_numpy_dispatch() -> None:
    """Tests the corresponding tensor functionality."""

    def my_np_func(x):
        """Test the corresponding tensor functionality."""
        return np.add(x, np.array([2.0], dtype=np.float32))

    x = Tensor((2,), DType.FLOAT32, "x")
    builder = trace(my_np_func, x)
    assert len(builder.nodes) == 1
    assert builder.nodes[0].op_type == "Add"


def test_dynamic_axes_trace() -> None:
    """Tests the corresponding tensor functionality."""
    from onnx9000.converters.frontend.tracer import trace

    def my_func(x):
        """Test the corresponding tensor functionality."""
        return x * 2.0

    x = Tensor(("N", 3, 224, 224), DType.FLOAT32, "x")
    builder = trace(my_func, x)
    assert builder.inputs[0].shape[0] == "N"
