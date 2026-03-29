"""Module docstring."""

import pytest
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.tflite_exporter.tf_protobuf import TFProtobufEncoder, SavedModelGenerator
import struct


def test_tf_protobuf_generation():
    """Provides functional implementation."""
    graph = Graph("TestGraph")
    graph.tensors["X"] = Tensor("X", shape=(1, 10), dtype="float32", is_initializer=False)

    w_data = struct.pack(f"<{100}f", *([1.0] * 100))
    graph.tensors["W"] = Tensor(
        "W", shape=(10, 10), dtype="float32", is_initializer=True, data=w_data
    )
    graph.tensors["W_int32"] = Tensor(
        "W_int32", shape=(1,), dtype="int32", is_initializer=True, data=b"1234"
    )
    graph.tensors["W_int64"] = Tensor(
        "W_int64", shape=(1,), dtype="int64", is_initializer=True, data=b"12345678"
    )
    graph.tensors["W_str"] = Tensor(
        "W_str", shape=(1,), dtype="string", is_initializer=True, data=b"str"
    )

    graph.nodes.append(Node("Add", ["X", "W"], ["Y"], name="add1"))
    graph.nodes.append(Node("Mul", ["Y", "W"], ["Z"], name="mul1"))
    graph.nodes.append(Node("Relu", ["Z"], ["Out"], name="relu1"))
    graph.nodes.append(Node("Unknown", ["X"], ["UnknownOut"], name="custom"))

    generator = SavedModelGenerator()
    saved_model = generator.generate_from_onnx(graph)

    assert saved_model["savedModelSchemaVersion"] == 1
    assert len(saved_model["metaGraphs"]) == 1

    meta_graph = saved_model["metaGraphs"][0]
    assert "serve" in meta_graph["metaInfoDef"]["tags"]

    nodes = meta_graph["graphDef"]["node"]
    assert len(nodes) == 8  # 4 const + 4 ops

    const_node = next(n for n in nodes if n["op"] == "Const")
    assert const_node is not None
    assert const_node["name"] == "W"

    add_node = next(n for n in nodes if n["name"] == "add1")
    assert add_node["op"] == "AddV2"

    custom_node = next(n for n in nodes if n["name"] == "custom")
    assert custom_node["op"] == "Custom_Unknown"

    encoder = TFProtobufEncoder()
    buf = encoder.encode(saved_model)

    assert len(buf) > 0
    assert buf[0] == 0x0A
