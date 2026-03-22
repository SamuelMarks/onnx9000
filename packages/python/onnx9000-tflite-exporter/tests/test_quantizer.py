import pytest
import struct
from onnx9000.core.ir import Graph, Tensor, ValueInfo
from onnx9000.tflite_exporter.quantization.quantizer import Quantizer


def test_quantizer_fp16():
    graph = Graph("test")
    # 1.0, -1.0, 0.0, 2.0, inf, NaN, tiny
    f32_data = struct.pack("<7f", 1.0, -1.0, 0.0, 2.0, 1000000.0, 0.000001, -1000000.0)
    graph.tensors["W"] = Tensor(
        "W", shape=(7,), dtype="float32", is_initializer=True, data=f32_data
    )
    graph.tensors["X"] = Tensor("X", shape=(4,), dtype="float32", is_initializer=False)
    graph.value_info.append(ValueInfo("X", (4,), "float32"))

    quantizer = Quantizer(graph, mode="fp16")
    quantizer.quantize()

    assert graph.tensors["W"].dtype == "float16"
    assert len(graph.tensors["W"].data) == 14

    # Test FP16 conversion values (approx)
    # 1.0 in fp16 is 0x3C00
    # -1.0 in fp16 is 0xBC00
    # 0.0 in fp16 is 0x0000
    # 2.0 in fp16 is 0x4000
    unpacked = struct.unpack("<7H", graph.tensors["W"].data)
    assert unpacked[0] == 0x3C00
    assert unpacked[1] == 0xBC00
    assert unpacked[2] == 0x0000
    assert unpacked[3] == 0x4000


def test_quantizer_int8():
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
    quantizer.quantize()  # Should parse

    assert "Y" in quantizer.quantization_map
    q = quantizer.quantization_map["Y"]
    assert q.scale == [0.5, 0.5]
    assert q.zero_point == [-5, -5]
    assert q.quantized_dimension == 1

    # Test flatbuffer extraction
    from onnx9000.tflite_exporter.flatbuffer.builder import FlatBufferBuilder

    builder = FlatBufferBuilder(1024)
    offset = quantizer.get_quantization_offset(builder, Tensor("Y"))
    assert offset > 0


def test_quantizer_none():
    graph = Graph("test")
    graph.tensors["W"] = Tensor("W", shape=(4,), dtype="float32", is_initializer=True, data=b"1234")
    quantizer = Quantizer(graph, mode="none")
    quantizer.quantize()
    assert graph.tensors["W"].dtype == "float32"  # unchanged


def test_qdq_quantization_extraction():
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

    # 3. QDQ scales with Relu
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

    # Per-Channel Warning
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
    graph.nodes.append(
        Node(
            "Conv",
            ["Y_quant2", "W_quant"],
            ["Y_quant3"],
            {},
            "conv_none",
        )
    )

    quantizer = Quantizer(graph, "int8")
    quantizer.quantize()

    class MockBuilder:
        def start_object(self, n):
            pass

        def add_field_offset(self, f, v, d):
            pass

        def add_field_int8(self, f, v, d):
            pass

        def add_field_int32(self, f, v, d):
            pass

        def end_object(self):
            return 42

        def create_float32_vector(self, v):
            return 1

        def create_int64_vector(self, v):
            return 2

        def start_vector(self, e, c, a):
            pass

        def add_float32(self, v):
            pass

        def add_int32(self, v):
            pass

        def end_vector(self, l):
            return 3

    offset = quantizer.get_quantization_offset(
        MockBuilder(), Tensor("X_quant", shape=(1,), dtype="int8", is_initializer=False)
    )
    assert offset == 42

    offset2 = quantizer.get_quantization_offset(
        MockBuilder(), Tensor("Unknown", shape=(1,), dtype="int8", is_initializer=False)
    )
    assert offset2 == 0
