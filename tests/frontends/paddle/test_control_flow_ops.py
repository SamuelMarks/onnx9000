import pytest
from onnx9000.frontends.paddle.builder import PaddleToONNXGraphBuilder
from onnx9000.frontends.paddle.parsers import PaddleNode
from onnx9000.frontends.paddle.control_flow_ops import CONTROL_FLOW_OPS_MAPPING


def test_paddle_cf_ops():
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "conditional_block", inputs={"Cond": ["c"]})
    outs = CONTROL_FLOW_OPS_MAPPING["conditional_block"](builder, n)
    assert builder.graph.nodes[-1].op_type == "If"
    n = PaddleNode("n", "while", inputs={"X": ["a"]})
    outs = CONTROL_FLOW_OPS_MAPPING["while"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Loop"
    for op, onnx_op in [("rnn", "RNN"), ("lstm", "LSTM"), ("gru", "GRU")]:
        n = PaddleNode("n", op, inputs={"Input": ["a"]})
        outs = CONTROL_FLOW_OPS_MAPPING[op](builder, n)
        assert builder.graph.nodes[-1].op_type == onnx_op
    for op, onnx_op in [
        ("tensor_array_to_tensor", "ConcatFromSequence"),
        ("lod_array_length", "SequenceLength"),
        ("write_to_array", "SequenceInsert"),
        ("read_from_array", "SequenceAt"),
    ]:
        n = PaddleNode("n", op, inputs={"X": ["a"], "Value": ["v"]})
        outs = CONTROL_FLOW_OPS_MAPPING[op](builder, n)
        assert builder.graph.nodes[-1].op_type == onnx_op
        assert builder.graph.nodes[-1].inputs == ["a", "v"]
