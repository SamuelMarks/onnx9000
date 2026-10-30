"""Comprehensive unit tests for onnx9000.onnx2c and onnx9000_onnx2c."""

import onnx9000_onnx2c
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.onnx2c import compile_to_c, run


def test_onnx2c_compile_and_run() -> None:
    """Verify run() and compile_to_c() generating C89 header and source strings."""
    assert run() == "[onnx2c] processed"
    assert onnx9000_onnx2c.run() == "[onnx2c] processed"

    g = Graph("c_net")
    g.add_tensor(Tensor("X", (1, 4), DType.FLOAT32))
    g.add_tensor(Tensor("Y", (1, 4), DType.FLOAT32))
    g.inputs.append(ValueInfo("X", (1, 4), DType.FLOAT32))
    g.outputs.append(ValueInfo("Y", (1, 4), DType.FLOAT32))
    g.add_node(Node("Relu", ["X"], ["Y"], name="relu0"))

    header, source = compile_to_c(g, prefix="test_")
    assert "#ifndef" in header
    assert "test_" in source
