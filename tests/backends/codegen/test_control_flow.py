"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.dtypes import DType
from onnx9000.backends.codegen.generator import Generator
from onnx9000.backends.codegen.ops.control_flow import generate_if, generate_loop


def test_control_flow_ops():
    """Provides semantic functionality and verification."""
    g = Graph("test_cf")
    t_cond = Tensor("cond", (1,), DType.BOOL)
    t_cond.buffer_id = 0
    t_out1 = Tensor("out1", (1,), DType.FLOAT32)
    t_out1.buffer_id = 1
    t_out2 = Tensor("out2", (1,), None)
    t_out2.buffer_id = 2
    t_max_trip = Tensor("max_trip_count", (1,), DType.INT64)
    t_max_trip.buffer_id = 3
    t_out3 = Tensor("out3", (1,), DType.FLOAT32)
    t_out3.buffer_id = 4
    g.add_tensor(t_cond)
    g.add_tensor(t_out1)
    g.add_tensor(t_out2)
    g.add_tensor(t_max_trip)
    g.add_tensor(t_out3)
    ctx = Generator(g)
    node_if = Node("If", inputs=["cond"], outputs=["out1", "out2"], attributes={})
    code_if = generate_if(node_if, ctx)
    assert "if (cond.data[0] != 0.0f)" in code_if
    assert "_arena[1].resize(1 * sizeof(float));" in code_if
    assert "onnx9000::Tensor<float> out1" in code_if
    assert "_arena[2].resize(1 * sizeof(float));" in code_if
    assert "onnx9000::Tensor<float> out2" in code_if
    node_loop = Node(
        "Loop", inputs=["max_trip_count", "cond"], outputs=["out3"], attributes={}
    )
    code_loop = generate_loop(node_loop, ctx)
    assert (
        "int64_t trip_count = static_cast<int64_t>(max_trip_count.data[0]);"
        in code_loop
    )
    assert "bool keep_going = (cond.data[0] != 0.0f);" in code_loop
    assert "_arena[4].resize(1 * sizeof(float));" in code_loop
    assert "onnx9000::Tensor<float> out3" in code_loop
