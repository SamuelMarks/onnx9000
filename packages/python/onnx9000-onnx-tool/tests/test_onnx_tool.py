"""Comprehensive unit tests for onnx9000.onnx_tool and onnx9000_onnx_tool."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.onnx_tool import ONNXTool, ProfilerResult


def test_onnx_tool_profile_and_process() -> None:
    """Verify ONNXTool.process() and ONNXTool.profile_graph()."""
    o = ONNXTool()
    assert o.process("test") == "ONNX Tool processed test"

    g = Graph("profile_test")
    g.add_tensor(Tensor("X", (1, 8), DType.FLOAT32))
    g.add_tensor(Tensor("Y", (1, 8), DType.FLOAT32))
    g.add_node(Node("Relu", ["X"], ["Y"]))

    res = o.profile_graph(g)
    assert isinstance(res, ProfilerResult)
    assert res.peak_activation_bytes == 32
