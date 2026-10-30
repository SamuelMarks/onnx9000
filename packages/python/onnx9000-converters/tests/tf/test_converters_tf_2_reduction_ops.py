"""Tests the reduction ops module functionality."""

from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.converters.tf.parsers import TFNode
from onnx9000.converters.tf.reduction_ops import REDUCTION_OPS_MAPPING


def test_reduction_ops_simple_reduce() -> None:
    """Tests the reduction ops simple reduce functionality."""
    builder = TFToONNXGraphBuilder()
    node = TFNode("n1", "Sum", inputs=["a", "axes"], attr={"keep_dims": True})
    REDUCTION_OPS_MAPPING["Sum"](builder, node)
    assert builder.graph.nodes[-1].op_type == "ReduceSum"
    assert builder.graph.nodes[-1].attributes["keepdims"] == 1
    node = TFNode("n2", "Mean", inputs=["a", "axes"])
    REDUCTION_OPS_MAPPING["Mean"](builder, node)
    assert builder.graph.nodes[-1].op_type == "ReduceMean"
    assert builder.graph.nodes[-1].attributes["keepdims"] == 0
    for tf_op, onnx_op in [
        ("Prod", "ReduceProd"),
        ("Max", "ReduceMax"),
        ("Min", "ReduceMin"),
        ("All", "ReduceMin"),
        ("Any", "ReduceMax"),
        ("ArgMax", "ArgMax"),
        ("ArgMin", "ArgMin"),
    ]:
        REDUCTION_OPS_MAPPING[tf_op](builder, TFNode(f"n_{tf_op}", tf_op, inputs=["a", "axes"]))
        assert builder.graph.nodes[-1].op_type == onnx_op


def test_reduction_ops_bincount_cum() -> None:
    """Tests the reduction ops bincount cum functionality."""
    builder = TFToONNXGraphBuilder()
    REDUCTION_OPS_MAPPING["Bincount"](builder, TFNode("n", "Bincount", inputs=["a", "b"]))
    assert builder.graph.nodes[-1].op_type == "Custom_Bincount"
    REDUCTION_OPS_MAPPING["Cumsum"](
        builder,
        TFNode("n", "Cumsum", inputs=["a", "b"], attr={"exclusive": True, "reverse": False}),
    )
    assert builder.graph.nodes[-1].op_type == "CumSum"
    assert builder.graph.nodes[-1].attributes["exclusive"] == 1
    assert builder.graph.nodes[-1].attributes["reverse"] == 0
    REDUCTION_OPS_MAPPING["Cumprod"](builder, TFNode("n", "Cumprod", inputs=["a", "b"]))
    assert builder.graph.nodes[-1].op_type == "Custom_Cumprod"


def test_reduction_ops_logicals() -> None:
    """Tests the reduction ops logicals functionality."""
    builder = TFToONNXGraphBuilder()
    logicals = [
        ("LogicalAnd", "And"),
        ("LogicalOr", "Or"),
        ("LogicalNot", "Not"),
        ("LogicalXor", "Xor"),
        ("Equal", "Equal"),
        ("Greater", "Greater"),
        ("GreaterEqual", "GreaterOrEqual"),
        ("Less", "Less"),
        ("LessEqual", "LessOrEqual"),
    ]
    for tf_op, onnx_op in logicals:
        REDUCTION_OPS_MAPPING[tf_op](builder, TFNode(f"n_{tf_op}", tf_op, inputs=["a", "b"]))
        assert builder.graph.nodes[-1].op_type == onnx_op
    REDUCTION_OPS_MAPPING["NotEqual"](builder, TFNode("n", "NotEqual", inputs=["a", "b"]))
    assert builder.graph.nodes[-1].op_type == "Not"
    assert builder.graph.nodes[-2].op_type == "Equal"
