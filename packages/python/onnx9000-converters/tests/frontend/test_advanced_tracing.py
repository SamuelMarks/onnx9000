"""Tests for advanced tracing with nested collections."""

from onnx9000.converters.frontend.nn.module import Module
from onnx9000.converters.frontend.tensor import Tensor
from onnx9000.converters.frontend.tracer import trace
from onnx9000.core.dtypes import DType


class NestedModule(Module):
    """Module that takes and returns nested collections."""

    def forward(self, x_list, y_dict):
        """Test the forward functionality."""
        out1 = x_list[0] + x_list[1]
        out2 = y_dict["a"] * 2
        return {"results": [out1, out2], "status": "ok"}


def test_nested_tracing():
    """Verify that trace() correctly handles nested lists and dicts."""
    m = NestedModule()
    x1 = Tensor((2, 2), DType.FLOAT32, "x1")
    x2 = Tensor((2, 2), DType.FLOAT32, "x2")
    y_a = Tensor((2, 2), DType.FLOAT32, "y_a")

    builder = trace(m, [x1, x2], {"a": y_a})

    # Verify inputs: should find x1, x2, and y_a
    input_names = [inp.name for inp in builder.inputs]
    assert "x1" in input_names
    assert "x2" in input_names
    assert "y_a" in input_names
    assert len(builder.inputs) == 3

    # Verify outputs: should find the two result tensors, ignoring the string "ok"
    assert len(builder.outputs) == 2
    for out in builder.outputs:
        assert isinstance(out, Tensor)


def test_deeply_nested_tracing():
    """Verify that trace() handles deeply nested structures."""

    def func(nested_in):
        return [nested_in["a"][0] + 1]

    x = Tensor((1,), DType.FLOAT32, "x")
    builder = trace(func, {"a": [x, "ignore"], "b": 123})

    assert len(builder.inputs) == 1
    assert builder.inputs[0].name == "x"
    assert len(builder.outputs) == 1
    assert len(builder.nodes) == 1
    assert builder.nodes[0].op_type == "Add"
