"""Module docstring."""

import struct

import pytest
from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.tflite_exporter.compiler.operators import ELEMENTWISE_OPS
from onnx9000.tflite_exporter.compiler.subgraph import compile_graph_to_tflite
from onnx9000.tflite_exporter.exporter import TFLiteExporter
from onnx9000.tflite_exporter.flatbuffer.reader import FlatBufferReader


def test_compiler_all_operators():
    """Provides functional implementation."""
    for op_type, mapping in ELEMENTWISE_OPS.items():
        exporter = TFLiteExporter()
        graph = Graph("TestOpGraph")

        graph.tensors["X"] = Tensor(
            "X", shape=(1, 10, 10, 3), dtype="float32", is_initializer=False
        )
        graph.tensors["Y"] = Tensor(
            "Y", shape=(1, 10, 10, 3), dtype="float32", is_initializer=False
        )
        graph.tensors["Z"] = Tensor(
            "Z", shape=(1, 10, 10, 3), dtype="float32", is_initializer=False
        )
        w_data = struct.pack(f"<{81}f", *([1.0] * 81))
        graph.tensors["W"] = Tensor(
            "W", shape=(3, 3, 3, 3), dtype="float32", is_initializer=True, data=w_data
        )

        graph.inputs.append(ValueInfo("X", (1, 10, 10, 3), "float32"))
        graph.outputs.append(ValueInfo("Z", (1, 10, 10, 3), "float32"))

        from onnx9000.core.ir import Attribute

        attrs = {}
        if op_type == "LeakyRelu":
            attrs["alpha"] = Attribute("alpha", "FLOAT", 0.1)
        if op_type == "LRN":
            attrs["size"] = Attribute("size", "INT", 3)
        if op_type in ["Concat", "Gather"]:
            attrs["axis"] = Attribute("axis", "INT", 1)
        if op_type in ["SpaceToDepth", "DepthToSpace"]:
            attrs["blocksize"] = Attribute("blocksize", "INT", 2)

        inputs = ["X"]
        outputs = ["Z"]

        if op_type in ["Add", "Sub", "Mul", "Div", "Equal", "Less", "Greater"]:
            inputs = ["X", "Y"]
        if op_type in ["Conv", "ConvTranspose", "Gemm", "MatMul"]:
            inputs = ["X", "W"]
        if op_type in ["Split", "SplitV", "SplitToSequence"]:
            outputs = ["Y", "Z"]

        node = Node(
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            attributes=attrs,
            name=f"{op_type}_node",
        )

        if op_type in ["Add", "Sub", "Mul", "Div"]:
            node.attributes["fused_activation"] = Attribute("fused_activation", "STRING", "Relu")
        if op_type == "CumSum":
            node.attributes["exclusive"] = Attribute("exclusive", "INT", 1)
            node.attributes["reverse"] = Attribute("reverse", "INT", 1)

        graph.nodes.append(node)

        subgraphs_offset = compile_graph_to_tflite(
            graph, exporter, keep_nchw=True, quant_mode="none"
        )
        exporter.builder.start_vector(4, 1, 4)
        exporter.builder.add_offset(subgraphs_offset)
        subgraphs_vec_offset = exporter.builder.end_vector(1)
        buf = exporter.finish(subgraphs_vec_offset, "test")

        reader = FlatBufferReader(buf)
        assert reader.check_magic_bytes("TFL3")

        model_offset = reader.get_root()
        op_codes_vec = reader.get_indirect_offset(model_offset, 1)
        assert reader.get_vector_length(op_codes_vec) == 1

        op_code_ptr = reader.bytes[op_codes_vec + 4 : op_codes_vec + 8]
        op_code_offset_val = struct.unpack("<I", op_code_ptr)[0]
        op_code_obj = op_codes_vec + 4 + op_code_offset_val

        code = reader.get_int8(op_code_obj, 0)
        extended_code = reader.get_int32(op_code_obj, 3, 0)
        final_code = extended_code if extended_code != 0 else (code + 256 if code < 0 else code)
        assert final_code == mapping.builtin_code


def test_compiler_conv_properties():
    """Provides functional implementation."""
    exporter = TFLiteExporter()
    graph = Graph("TestOpGraph")
    graph.tensors["X"] = Tensor("X", shape=(1, 10, 10, 3), dtype="float32", is_initializer=False)
    graph.tensors["Z"] = Tensor("Z", shape=(1, 10, 10, 3), dtype="float32", is_initializer=False)
    w_data = struct.pack(f"<{81}f", *([1.0] * 81))
    graph.tensors["W"] = Tensor(
        "W", shape=(3, 3, 3, 3), dtype="float32", is_initializer=True, data=w_data
    )
    graph.inputs.append(ValueInfo("X", (1, 10, 10, 3), "float32"))
    graph.outputs.append(ValueInfo("Z", (1, 10, 10, 3), "float32"))

    from onnx9000.core.ir import Attribute

    # Conv
    graph.nodes.append(
        Node(
            "Conv",
            ["X", "W"],
            ["Z"],
            {
                "strides": Attribute("strides", "INTS", [2, 2]),
                "dilations": Attribute("dilations", "INTS", [1, 1]),
            },
            "conv1",
        )
    )

    # Depthwise Conv
    graph.nodes.append(
        Node(
            "Conv",
            ["X", "W"],
            ["Z"],
            {
                "strides": Attribute("strides", "INTS", [2, 2]),
                "group": Attribute("group", "INT", 3),
            },
            "conv2",
        )
    )

    # MaxPool
    graph.nodes.append(
        Node(
            "MaxPool",
            ["X"],
            ["Z"],
            {
                "strides": Attribute("strides", "INTS", [2, 2]),
                "kernel_shape": Attribute("kernel_shape", "INTS", [3, 3]),
            },
            "pool1",
        )
    )

    # Custom
    graph.nodes.append(Node("NonMaxSuppression", ["X"], ["Z"], {}, "nms"))
    graph.nodes.append(Node("MyCustomOp", ["X"], ["Z"], {}, "custom", "org.test"))

    subgraphs_offset = compile_graph_to_tflite(graph, exporter, keep_nchw=True, quant_mode="none")
    exporter.builder.start_vector(4, 1, 4)
    exporter.builder.add_offset(subgraphs_offset)
    subgraphs_vec_offset = exporter.builder.end_vector(1)
    buf = exporter.finish(subgraphs_vec_offset, "test")
    assert len(buf) > 0
