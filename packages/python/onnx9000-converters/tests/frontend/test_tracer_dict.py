"""Unit tests and verification suite for tracer dict functionality."""

from typing import NoReturn

import pytest
from onnx9000.converters.frontend.nn.module import Module
from onnx9000.converters.frontend.tensor import Tensor
from onnx9000.converters.frontend.tracer import trace
from onnx9000.core.dtypes import DType
from onnx9000.core.exceptions import Onnx9000Error


class MyDictModule(Module):
    """Class MyDictModule implementation."""

    def forward(
        self, x: int | float | str | bool | bytes | list | dict | tuple | None
    ) -> int | float | str | bool | bytes | list | dict | tuple | None:
        """Test the forward functionality."""
        return {"out1": x + 1, "out2": x * 2}


def test_tracer_dict() -> None:
    """Tests the test_tracer_dict functionality."""
    m = MyDictModule()
    x = Tensor((2, 2), DType.FLOAT32, "x")
    builder = trace(m, x)
    assert len(builder.outputs) == 2


def test_tracer_fallback() -> None:
    """Tests the test_tracer_fallback functionality."""

    def err_func(x: int | float | str | bool | bytes | list | dict | tuple | None) -> NoReturn:
        """Test the err_func functionality."""
        raise Onnx9000Error("Simulated eager error")

    x = Tensor((2, 2), DType.FLOAT32, "x")
    with pytest.raises(RuntimeError) as excinfo:
        trace(err_func, x)
    assert "Tracing failed" in str(excinfo.value)
