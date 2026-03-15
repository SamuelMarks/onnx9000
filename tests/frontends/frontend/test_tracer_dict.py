"""Module providing core logic and structural definitions."""

from onnx9000.frontends.frontend.tracer import trace
from onnx9000.frontends.frontend.tensor import Tensor
from onnx9000.frontends.frontend.nn.module import Module
from onnx9000.core.dtypes import DType
import pytest


class MyDictModule(Module):
    """Provides semantic functionality and verification."""

    def forward(self, x):
        """Provides semantic functionality and verification."""
        return {"out1": x + 1, "out2": x * 2}


def test_tracer_dict():
    """Provides semantic functionality and verification."""
    m = MyDictModule()
    x = Tensor((2, 2), DType.FLOAT32, "x")
    builder = trace(m, x)
    assert len(builder.outputs) == 2


def test_tracer_fallback():
    """Provides semantic functionality and verification."""

    def err_func(x):
        """Provides semantic functionality and verification."""
        raise ValueError("Simulated eager error")

    x = Tensor((2, 2), DType.FLOAT32, "x")
    with pytest.raises(RuntimeError) as excinfo:
        trace(err_func, x)
    assert "Tracing failed" in str(excinfo.value)
