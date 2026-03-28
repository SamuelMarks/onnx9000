"""Tests the jit wrapper module functionality."""

import numpy as np
import pytest
from onnx9000.converters.jit.wrapper import CompiledModel
from onnx9000.core.ir import Graph


class MockCppModel:
    """Represents the Mock Cpp Model class."""

    def forward(self, *args):
        """Execute the forward operation."""
        if len(args) == 1:
            return args[0]
        return args


def test_compiled_model_wrapper() -> None:
    """Tests the compiled model wrapper functionality."""
    g = Graph("test")
    g.inputs.append("in1")
    g.inputs.append("in2")
    cpp = MockCppModel()
    model = CompiledModel(cpp, g)
    out = model(np.array([1]), [2])
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert isinstance(out[1], np.ndarray)
    g2 = Graph("single")
    g2.inputs.append("in1")
    model2 = CompiledModel(cpp, g2)
    out2 = model2([1])
    assert isinstance(out2, tuple)
    assert len(out2) == 1
    with pytest.raises(ValueError, match="Expected 2 inputs, got 1"):
        model([1])
