from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.converters.tf.control_flow_ops import CONTROL_FLOW_OPS_MAPPING
from onnx9000.converters.tf.parsers import TFNode


def test_control_flow_ops_noop() -> None:
    builder = TFToONNXGraphBuilder()
    for op in ["Enter", "Exit", "Merge", "Switch", "NextIteration", "LoopCond"]:
        CONTROL_FLOW_OPS_MAPPING[op](builder, TFNode(f"n_{op}", op, inputs=["a"]))
        assert builder.graph.nodes[-1].op_type == f"Custom_TF{op}"


def test_control_flow_ops_if_while() -> None:
    builder = TFToONNXGraphBuilder()
    CONTROL_FLOW_OPS_MAPPING["If"](
        builder,
        TFNode(
            "n_if", "If", inputs=["cond"], attr={"then_branch": "graph1", "else_branch": "graph2"}
        ),
    )
    assert builder.graph.nodes[-1].op_type == "If"
    assert builder.graph.nodes[-1].attributes["then_branch"] == "graph1"
    assert builder.graph.nodes[-1].attributes["else_branch"] == "graph2"
    CONTROL_FLOW_OPS_MAPPING["StatelessIf"](
        builder,
        TFNode(
            "n_if", "If", inputs=["cond"], attr={"then_branch": "graph1", "else_branch": "graph2"}
        ),
    )
    assert builder.graph.nodes[-1].op_type == "If"
    CONTROL_FLOW_OPS_MAPPING["While"](
        builder,
        TFNode("n_while", "While", inputs=["a", "b"], attr={"cond": "graph1", "body": "graph2"}),
    )
    assert builder.graph.nodes[-1].op_type == "Loop"
    assert builder.graph.nodes[-1].attributes["cond"] == "graph1"
    assert builder.graph.nodes[-1].attributes["body"] == "graph2"
    CONTROL_FLOW_OPS_MAPPING["StatelessWhile"](
        builder,
        TFNode("n_while", "While", inputs=["a", "b"], attr={"cond": "graph1", "body": "graph2"}),
    )
    assert builder.graph.nodes[-1].op_type == "Loop"


def test_control_flow_ops_tensor_array() -> None:
    builder = TFToONNXGraphBuilder()
    outs = CONTROL_FLOW_OPS_MAPPING["TensorArrayV3"](
        builder, TFNode("n_ta", "TensorArrayV3", inputs=["size"], attr={"dtype": 2})
    )
    assert len(outs) == 2
    assert builder.graph.nodes[-1].op_type == "SequenceEmpty"
    assert builder.graph.nodes[-1].attributes["dtype"] == 2
    assert builder.graph.tensors[outs[1]].shape == ()
    outs = CONTROL_FLOW_OPS_MAPPING["TensorArrayReadV3"](
        builder, TFNode("n_read", "TensorArrayReadV3", inputs=["handle", "index", "flow"])
    )
    assert builder.graph.nodes[-1].op_type == "SequenceAt"
    outs = CONTROL_FLOW_OPS_MAPPING["TensorArrayWriteV3"](
        builder,
        TFNode("n_write", "TensorArrayWriteV3", inputs=["handle", "index", "value", "flow"]),
    )
    assert builder.graph.nodes[-1].op_type == "SequenceInsert"
    assert builder.graph.nodes[-1].inputs == ["handle", "value", "index"]
    outs = CONTROL_FLOW_OPS_MAPPING["TensorArraySizeV3"](
        builder, TFNode("n_size", "TensorArraySizeV3", inputs=["handle", "flow"])
    )
    assert builder.graph.nodes[-1].op_type == "SequenceLength"
    outs = CONTROL_FLOW_OPS_MAPPING["TensorArrayGatherV3"](
        builder, TFNode("n_gather", "TensorArrayGatherV3", inputs=["handle", "indices", "flow"])
    )
    assert builder.graph.nodes[-1].op_type == "ConcatFromSequence"
    assert builder.graph.nodes[-1].attributes["axis"] == 0
    assert builder.graph.nodes[-1].attributes["new_axis"] == 1
    outs = CONTROL_FLOW_OPS_MAPPING["TensorArrayScatterV3"](
        builder,
        TFNode("n_scatter", "TensorArrayScatterV3", inputs=["handle", "indices", "value", "flow"]),
    )
    assert builder.graph.nodes[-1].op_type == "SplitToSequence"
    assert builder.graph.nodes[-1].inputs == ["value"]
    assert builder.graph.nodes[-1].attributes["axis"] == 0
    assert builder.graph.nodes[-1].attributes["keepdims"] == 0


def test_control_flow_ops_rnn() -> None:
    builder = TFToONNXGraphBuilder()
    for op in ["BasicLSTMCell", "LSTMBlockCell", "BlockLSTM"]:
        CONTROL_FLOW_OPS_MAPPING[op](builder, TFNode(f"n_{op}", op, inputs=["a"]))
        assert builder.graph.nodes[-1].op_type == "LSTM"
    CONTROL_FLOW_OPS_MAPPING["GRUBlockCell"](builder, TFNode("n_gru", "GRUBlockCell", inputs=["a"]))
    assert builder.graph.nodes[-1].op_type == "GRU"
