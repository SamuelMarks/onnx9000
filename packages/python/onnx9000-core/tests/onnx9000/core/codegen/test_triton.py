"""Unit tests for Triton kernel exporter in `onnx9000.core.codegen.triton`."""

from onnx9000.core.codegen.triton import TritonExporter
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, ValueInfo


def test_triton_exporter() -> None:
    """Verify TritonExporter generates Triton kernel source code."""
    g = Graph("triton_test")
    g.inputs.append(ValueInfo("x", (1, 4), DType.FLOAT32))
    g.outputs.append(ValueInfo("y", (1, 4), DType.FLOAT32))
    g.add_node(Node("Relu", ["x"], ["y"], name="r0"))

    code = TritonExporter(g).export()
    assert "import triton" in code
