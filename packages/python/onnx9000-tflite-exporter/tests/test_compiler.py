"""Module docstring."""

import struct

from onnx9000.core.ir import Graph, Node, Tensor, ValueInfo
from onnx9000.tflite_exporter.compiler.subgraph import compile_graph_to_tflite
from onnx9000.tflite_exporter.exporter import TFLiteExporter
from onnx9000.tflite_exporter.flatbuffer.reader import FlatBufferReader
from onnx9000.tflite_exporter.flatbuffer.schema import BuiltinOperator


def test_compiler_subgraph_mapping():
    """Provides functional implementation."""
    exporter = TFLiteExporter()
    graph = Graph("TestGraph")

    # Add some tensors
    t_input1 = Tensor("input1", shape=(1, 3, 224, 224), dtype="float32", is_initializer=False)
    t_output1 = Tensor("output1", shape=(1, 1000), dtype="float32", is_initializer=False)

    w_data = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    t_weight1 = Tensor(
        "weight1", shape=(1000, 3, 3, 3), dtype="float32", is_initializer=True, data=w_data
    )

    graph.tensors["input1"] = t_input1
    graph.tensors["output1"] = t_output1
    graph.tensors["weight1"] = t_weight1

    graph.inputs.append(ValueInfo("input1", (1, 3, 224, 224), "float32"))
    graph.outputs.append(ValueInfo("output1", (1, 1000), "float32"))

    subgraphs_offset = compile_graph_to_tflite(graph, exporter, keep_nchw=True)

    exporter.builder.start_vector(4, 1, 4)
    exporter.builder.add_offset(subgraphs_offset)
    subgraphs_vec_offset = exporter.builder.end_vector(1)

    buf = exporter.finish(subgraphs_vec_offset, "test_graph_compilation")

    reader = FlatBufferReader(buf)
    assert reader.check_magic_bytes("TFL3")

    model_offset = reader.get_root()
    subgraphs_vec = reader.get_indirect_offset(model_offset, 2)

    assert subgraphs_vec != 0
    num_subgraphs = reader.get_vector_length(subgraphs_vec)
    assert num_subgraphs == 1

    # In Python, we have get_indirect_offset which offsets directly from the vector item
    # Wait, get_indirect_offset takes vtable index. Here we need an array offset.
    # We will manually dereference array elements.
    subgraph_ptr = reader.bytes[subgraphs_vec + 4 : subgraphs_vec + 8]
    subgraph_offset_val = struct.unpack("<I", subgraph_ptr)[0]
    subgraph_loc = subgraphs_vec + 4 + subgraph_offset_val

    # Subgraph inputs is field 1
    inputs_vec = reader.get_indirect_offset(subgraph_loc, 1)
    assert reader.get_vector_length(inputs_vec) == 1

    # Subgraph outputs is field 2
    outputs_vec = reader.get_indirect_offset(subgraph_loc, 2)
    assert reader.get_vector_length(outputs_vec) == 1

    # Subgraph tensors is field 0
    tensors_vec = reader.get_indirect_offset(subgraph_loc, 0)
    assert reader.get_vector_length(tensors_vec) == 3


def test_compiler_elementwise():
    """Test the compilation of elementwise operations to TFLite."""
    exporter = TFLiteExporter()
    graph = Graph("TestGraph")

    # Add some tensors
    graph.tensors["A"] = Tensor("A", shape=(1, 10), dtype="float32", is_initializer=False)
    graph.tensors["B"] = Tensor("B", shape=(1, 10), dtype="float32", is_initializer=False)
    graph.tensors["C"] = Tensor("C", shape=(1, 10), dtype="float32", is_initializer=False)

    graph.inputs.append(ValueInfo("A", (1, 10), "float32"))
    graph.inputs.append(ValueInfo("B", (1, 10), "float32"))
    graph.outputs.append(ValueInfo("C", (1, 10), "float32"))

    node = Node(op_type="Add", inputs=["A", "B"], outputs=["C"], name="add1")
    graph.nodes.append(node)

    subgraphs_offset = compile_graph_to_tflite(graph, exporter, keep_nchw=True)

    exporter.builder.start_vector(4, 1, 4)
    exporter.builder.add_offset(subgraphs_offset)
    subgraphs_vec_offset = exporter.builder.end_vector(1)

    buf = exporter.finish(subgraphs_vec_offset, "test_add")
    reader = FlatBufferReader(buf)

    model_offset = reader.get_root()

    # Check opcodes
    op_codes_vec = reader.get_indirect_offset(model_offset, 1)
    assert reader.get_vector_length(op_codes_vec) == 1

    op_code_ptr = reader.bytes[op_codes_vec + 4 : op_codes_vec + 8]
    op_code_offset_val = struct.unpack("<I", op_code_ptr)[0]
    op_code_obj = op_codes_vec + 4 + op_code_offset_val

    builtin_code = reader.get_int8(op_code_obj, 0)
    assert builtin_code == BuiltinOperator.ADD

    # Check operators
    subgraphs_vec = reader.get_indirect_offset(model_offset, 2)
    subgraph_ptr = reader.bytes[subgraphs_vec + 4 : subgraphs_vec + 8]
    subgraph_offset_val = struct.unpack("<I", subgraph_ptr)[0]
    subgraph_loc = subgraphs_vec + 4 + subgraph_offset_val

    operators_vec = reader.get_indirect_offset(subgraph_loc, 3)
    assert reader.get_vector_length(operators_vec) == 1

    op_ptr = reader.bytes[operators_vec + 4 : operators_vec + 8]
    op_offset_val = struct.unpack("<I", op_ptr)[0]
    op_obj = operators_vec + 4 + op_offset_val

    op_idx = reader.get_int32(op_obj, 0)
    assert op_idx == 0

    inputs_vec = reader.get_indirect_offset(op_obj, 1)
    assert reader.get_vector_length(inputs_vec) == 2

    outputs_vec = reader.get_indirect_offset(op_obj, 2)
    assert reader.get_vector_length(outputs_vec) == 1
