"""Module providing core logic and structural definitions."""

from onnx9000.frontends.frontend.tracer import Tracer, Proxy, trace
from onnx9000.frontends.frontend.tensor import Tensor
from onnx9000.core.dtypes import DType
import pytest


def test_tracer_and_proxy():
    """Provides semantic functionality and verification."""

    def my_func(x, y):
        """Provides semantic functionality and verification."""
        return x + y

    x = Tensor((2, 2), DType.FLOAT32)
    y = Tensor((2, 2), DType.FLOAT32)
    builder = trace(my_func, x, y)
    assert len(builder.nodes) == 1
    assert builder.nodes[0].op_type == "Add"
    assert len(builder.inputs) == 2
    assert len(builder.outputs) == 1
    assert isinstance(builder.inputs[0], Tensor)


def test_tracer_constant_folding():
    """Provides semantic functionality and verification."""

    def const_func(x):
        """Provides semantic functionality and verification."""
        return x + 2

    x = Tensor((2, 2), DType.FLOAT32)
    builder = trace(const_func, x)
    assert len(builder.nodes) == 1
    assert len(builder.parameters) == 1
    assert builder.parameters[0].name.startswith("constant_")


def test_tracer_control_flow():
    """Provides semantic functionality and verification."""

    def dynamic_if(x):
        """Provides semantic functionality and verification."""
        if x > 0:
            return x + 1
        return x - 1

    x = Tensor((2, 2), DType.FLOAT32)
    with pytest.raises(RuntimeError) as excinfo:
        trace(dynamic_if, x)
    assert "Data-dependent control flow is not supported" in str(excinfo.value)
