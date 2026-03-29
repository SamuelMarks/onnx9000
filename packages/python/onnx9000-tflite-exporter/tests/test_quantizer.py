"""Tests for packages/python/onnx9000-tflite-exporter/tests/test_quantizer.py."""

import pytest
import struct
from onnx9000.core.ir import Graph, Tensor, ValueInfo
from onnx9000.tflite_exporter.quantization.quantizer import Quantizer


def test_quantizer_fp16():
    """Test quantizer fp16."""
    graph = Graph("test")
    f32_data = struct.pack("<7f", 1.0, -1.0, 0.0, 2.0, 1000000.0, 1e-06, -1000000.0)
    graph.tensors["W"] = Tensor(
        "W", shape=(7,), dtype="float32", is_initializer=True, data=f32_data
    )
    graph.tensors["X"] = Tensor("X", shape=(4,), dtype="float32", is_initializer=False)
    graph.value_info.append(ValueInfo("X", (4,), "float32"))
    quantizer = Quantizer(graph, mode="fp16")
    quantizer.quantize()
    assert graph.tensors["W"].dtype == "float16"
    assert len(graph.tensors["W"].data) == 14
    unpacked = struct.unpack("<7H", graph.tensors["W"].data)
    assert unpacked[0] == 15360
    assert unpacked[1] == 48128
    assert unpacked[2] == 0
    assert unpacked[3] == 16384


def test_quantizer_int8():
    """Test quantizer int8."""
    from onnx9000.core.ir import Node, Attribute

    graph = Graph("test")
    graph.tensors["Scale"] = Tensor(
        "Scale", shape=(2,), dtype="float32", is_initializer=True, data=struct.pack("<2f", 0.5, 0.5)
    )
    graph.tensors["ZP"] = Tensor(
        "ZP", shape=(2,), dtype="int8", is_initializer=True, data=struct.pack("<2b", -5, -5)
    )
    graph.nodes.append(
        Node("QuantizeLinear", ["X", "Scale", "ZP"], ["Y"], {"axis": Attribute("axis", "INT", 1)})
    )
    quantizer = Quantizer(graph, mode="int8")
    quantizer.quantize()
    assert "Y" in quantizer.quantization_map
    q = quantizer.quantization_map["Y"]
    assert q.scale == [0.5, 0.5]
    assert q.zero_point == [-5, -5]
    assert q.quantized_dimension == 1
    from onnx9000.tflite_exporter.flatbuffer.builder import FlatBufferBuilder

    builder = FlatBufferBuilder(1024)
    offset = quantizer.get_quantization_offset(builder, Tensor("Y"))
    assert offset > 0


def test_quantizer_none():
    """Test quantizer none."""
    graph = Graph("test")
    graph.tensors["W"] = Tensor("W", shape=(4,), dtype="float32", is_initializer=True, data=b"1234")
    quantizer = Quantizer(graph, mode="none")
    quantizer.quantize()
    assert graph.tensors["W"].dtype == "float32"


def test_qdq_quantization_extraction():
    """Test qdq quantization extraction."""
    from onnx9000.core.ir import Graph, Node, Tensor, Attribute
    from onnx9000.tflite_exporter.quantization.quantizer import Quantizer
    import struct

    graph = Graph("TestGraph")
    graph.tensors["scale"] = Tensor(
        "scale", shape=(1,), dtype="float32", is_initializer=True, data=struct.pack("<f", 0.5)
    )
    graph.tensors["zp"] = Tensor(
        "zp", shape=(1,), dtype="uint8", is_initializer=True, data=struct.pack("<B", 128)
    )
    graph.tensors["scale2"] = Tensor(
        "scale2", shape=(1,), dtype="float32", is_initializer=True, data=struct.pack("<f", 0.5)
    )
    graph.tensors["zp2"] = Tensor(
        "zp2", shape=(1,), dtype="int16", is_initializer=True, data=struct.pack("<h", 128)
    )
    graph.nodes.append(
        Node(
            "QuantizeLinear",
            ["X", "scale", "zp"],
            ["X_quant"],
            {"axis": Attribute("axis", "INT", 1)},
            "q1",
        )
    )
    graph.nodes.append(
        Node(
            "QuantizeLinear",
            ["X", "scale2", "zp2"],
            ["Y_quant"],
            {"axis": Attribute("axis", "INT", 1)},
            "q2",
        )
    )
    graph.tensors["scale3"] = Tensor(
        "scale3", shape=(1,), dtype="float32", is_initializer=True, data=struct.pack("<f", 0.5)
    )
    graph.tensors["zp3"] = Tensor(
        "zp3", shape=(1,), dtype="uint8", is_initializer=True, data=struct.pack("<B", 128)
    )
    graph.nodes.append(
        Node(
            "QuantizeLinear",
            ["X", "scale3", "zp3"],
            ["Y_quant2"],
            {"axis": Attribute("axis", "INT", 1)},
            "q3",
        )
    )
    graph.tensors["scale_per_channel"] = Tensor(
        "scale_pc",
        shape=(2,),
        dtype="float32",
        is_initializer=True,
        data=struct.pack("<2f", 0.5, 0.5),
    )
    graph.tensors["zp_per_channel"] = Tensor(
        "zp_pc", shape=(2,), dtype="uint8", is_initializer=True, data=struct.pack("<2B", 128, 128)
    )
    graph.nodes.append(
        Node(
            "DynamicQuantizeLinear",
            ["X", "scale_per_channel", "zp_per_channel"],
            ["X_quant_pc"],
            {},
            "q4_pc",
        )
    )
    graph.nodes.append(
        Node(
            "Conv",
            ["X_quant", "W_quant"],
            ["Y_quant_relu6"],
            {"fused_activation": Attribute("fused_activation", "STRING", "Relu6")},
            "conv_relu6",
        )
    )
    graph.nodes.append(
        Node(
            "Conv",
            ["Y_quant", "W_quant"],
            ["Y_quant_relu"],
            {"fused_activation": Attribute("fused_activation", "STRING", "Relu")},
            "conv_relu",
        )
    )
    graph.nodes.append(Node("Conv", ["Y_quant2", "W_quant"], ["Y_quant3"], {}, "conv_none"))
    quantizer = Quantizer(graph, "int8")
    quantizer.quantize()

    class MockBuilder:
        """MockBuilder implementation."""

        def start_object(self, n):
            """Perform start object operation."""
            return None

        def add_field_offset(self, f, v, d):
            """Perform add field offset operation."""
            return None

        def add_field_int8(self, f, v, d):
            """Perform add field int8 operation."""
            return None

        def add_field_int32(self, f, v, d):
            """Perform add field int32 operation."""
            return None

        def end_object(self):
            """Perform end object operation."""
            return 42

        def create_float32_vector(self, v):
            """Perform create float32 vector operation."""
            return 1

        def create_int64_vector(self, v):
            """Perform create int64 vector operation."""
            return 2

        def start_vector(self, e, c, a):
            """Perform start vector operation."""
            return None

        def add_float32(self, v):
            """Perform add float32 operation."""
            return None

        def add_int32(self, v):
            """Perform add int32 operation."""
            return None

        def end_vector(self, l):
            """Perform end vector operation."""
            return 3

    m = MockBuilder()
    m.create_float32_vector([1.0])
    m.create_int64_vector([1])
    offset = quantizer.get_quantization_offset(
        m, Tensor("X_quant", shape=(1,), dtype="int8", is_initializer=False)
    )
    assert offset == 42
    offset2 = quantizer.get_quantization_offset(
        MockBuilder(), Tensor("Unknown", shape=(1,), dtype="int8", is_initializer=False)
    )
    assert offset2 == 0
