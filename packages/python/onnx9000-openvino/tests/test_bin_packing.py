"""Tests for bin packing."""

import struct

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.openvino.exporter import OpenVinoExporter


def test_deduplication():
    """Docstring for D103."""
    graph = Graph("test_dedup")
    graph.inputs.append(ValueInfo("X", (1, 3, 224, 224), DType.FLOAT32))

    # Create two identical weight tensors
    w1_data = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    w1 = Tensor("W1", shape=(4,), dtype=DType.FLOAT32, is_initializer=True, data=w1_data)
    graph.tensors["W1"] = w1
    graph.initializers.append("W1")

    w2 = Tensor("W2", shape=(4,), dtype=DType.FLOAT32, is_initializer=True, data=w1_data)
    graph.tensors["W2"] = w2
    graph.initializers.append("W2")

    # Just to add to graph outputs
    graph.outputs.append(ValueInfo("Y", (1,), DType.FLOAT32))
    node = Node("Add", inputs=["W1", "W2"], outputs=["Y"])
    graph.nodes.append(node)

    exporter = OpenVinoExporter(graph)
    xml_str, bin_data = exporter.export()

    # Total size should be 16 bytes for one tensor, instead of 32
    assert len(bin_data) == 16

    # In XML, both should have same offset
    assert xml_str.count('offset="0"') >= 2


def test_fp16_cast():
    """Docstring for D103."""
    graph = Graph("test_fp16")
    w1_data = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    w1 = Tensor("W1", shape=(4,), dtype=DType.FLOAT32, is_initializer=True, data=w1_data)
    graph.tensors["W1"] = w1
    graph.initializers.append("W1")

    exporter = OpenVinoExporter(graph, compress_to_fp16=True)
    xml_str, bin_data = exporter.export()

    # Float32 is 16 bytes, Float16 is 8 bytes
    assert len(bin_data) == 8
    # Element type should be f16
    assert 'element_type="f16"' in xml_str
