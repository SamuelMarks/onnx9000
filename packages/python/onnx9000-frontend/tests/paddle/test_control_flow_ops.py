from onnx9000.frontend.paddle.builder import PaddleToONNXGraphBuilder
from onnx9000.frontend.paddle.control_flow_ops import CONTROL_FLOW_OPS_MAPPING
from onnx9000.frontend.paddle.parsers import PaddleNode


def test_paddle_cf_ops() -> None:
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "conditional_block", inputs={"Cond": ["c"]})
    CONTROL_FLOW_OPS_MAPPING["conditional_block"](builder, n)
    assert builder.graph.nodes[-1].op_type == "If"
    n = PaddleNode("n", "while", inputs={"X": ["a"]})
    CONTROL_FLOW_OPS_MAPPING["while"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Loop"
    for op, onnx_op in [("rnn", "RNN"), ("lstm", "LSTM"), ("gru", "GRU")]:
        n = PaddleNode("n", op, inputs={"Input": ["a"]})
        CONTROL_FLOW_OPS_MAPPING[op](builder, n)
        assert builder.graph.nodes[-1].op_type == onnx_op
    for op, onnx_op in [
        ("tensor_array_to_tensor", "ConcatFromSequence"),
        ("lod_array_length", "SequenceLength"),
        ("write_to_array", "SequenceInsert"),
        ("read_from_array", "SequenceAt"),
    ]:
        n = PaddleNode("n", op, inputs={"X": ["a"], "Value": ["v"]})
        CONTROL_FLOW_OPS_MAPPING[op](builder, n)
        assert builder.graph.nodes[-1].op_type == onnx_op
        assert builder.graph.nodes[-1].inputs == ["a", "v"]


def test_control_flow_ops_custom() -> None:
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "select_input", inputs={"X": ["a"]})
    CONTROL_FLOW_OPS_MAPPING["select_input"](builder, n)
    assert builder.graph.nodes[-1].op_type == "If"
