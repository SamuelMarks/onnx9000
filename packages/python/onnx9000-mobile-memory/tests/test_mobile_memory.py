"""Comprehensive unit tests for onnx9000.mobile_memory and onnx9000_mobile_memory."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.mobile_memory import MobileMemory, MobileMemoryReport


def test_mobile_memory_analyze_and_process() -> None:
    """Verify MobileMemory.process() and MobileMemory.analyze() on a multi-node Graph."""
    mm = MobileMemory()
    assert mm.process("test") == "Mobile Memory processed test"

    g = Graph("mobile_net")
    g.add_tensor(Tensor("X", (1, 16), DType.FLOAT32))
    g.add_tensor(Tensor("Y", (1, 16), DType.FLOAT32))
    g.inputs.append("X")
    g.outputs.append("Y")
    g.add_node(Node("Add", ["X", "X"], ["Y"]))

    report = mm.analyze(g, strategy="best_fit")
    assert isinstance(report, MobileMemoryReport)
    assert report.peak_bytes > 0
    assert report.strategy == "best_fit"
    assert "X" in report.tensor_offsets
