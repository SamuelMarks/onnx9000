from onnx9000.converters.paddle.builder import PaddleToONNXGraphBuilder
from onnx9000.converters.paddle.parsers import PaddleNode
from onnx9000.converters.paddle.reduction_ops import REDUCTION_OPS_MAPPING


def test_paddle_reduce_ops() -> None:
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "reduce_sum", inputs={"X": ["a"]}, attrs={"dim": [1], "keep_dim": True})
    outs = REDUCTION_OPS_MAPPING["reduce_sum"](builder, n)
    assert builder.graph.nodes[-1].op_type == "ReduceSum"
    assert builder.graph.nodes[-1].attributes["keepdims"] == 1
    assert "n_axes" in builder.graph.nodes[-1].inputs[-1]
    n2 = PaddleNode("n", "reduce_max", inputs={"X": ["a"]})
    outs = REDUCTION_OPS_MAPPING["reduce_max"](builder, n2)
    assert builder.graph.nodes[-1].op_type == "ReduceMax"
    assert builder.graph.nodes[-1].attributes["keepdims"] == 0
    for op in ["reduce_all", "reduce_any", "reduce_min", "reduce_prod", "reduce_mean"]:
        n = PaddleNode("n", op, inputs={"X": ["a"]})
        outs = REDUCTION_OPS_MAPPING[op](builder, n)
        assert len(outs) == 1


def test_paddle_arg_ops() -> None:
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "arg_max", inputs={"X": ["a"]}, attrs={"axis": 1, "keepdims": True})
    REDUCTION_OPS_MAPPING["arg_max"](builder, n)
    assert builder.graph.nodes[-1].op_type == "ArgMax"
    assert builder.graph.nodes[-1].attributes["axis"] == 1
    assert builder.graph.nodes[-1].attributes["keepdims"] == 1
    n2 = PaddleNode("n", "arg_min", inputs={"X": ["a"]})
    REDUCTION_OPS_MAPPING["arg_min"](builder, n2)
    assert builder.graph.nodes[-1].op_type == "ArgMin"


def test_paddle_cumsum() -> None:
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode(
        "n", "cumsum", inputs={"X": ["a"]}, attrs={"axis": 1, "exclusive": True, "reverse": True}
    )
    REDUCTION_OPS_MAPPING["cumsum"](builder, n)
    assert builder.graph.nodes[-1].op_type == "CumSum"
    assert builder.graph.nodes[-1].attributes["exclusive"] == 1
    assert builder.graph.nodes[-1].attributes["reverse"] == 1


def test_paddle_logical_ops() -> None:
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "logical_and", inputs={"X": ["a"], "Y": ["b"]})
    outs = REDUCTION_OPS_MAPPING["logical_and"](builder, n)
    assert builder.graph.nodes[-1].op_type == "And"
    n2 = PaddleNode("n", "logical_not", inputs={"X": ["a"]})
    outs = REDUCTION_OPS_MAPPING["logical_not"](builder, n2)
    assert builder.graph.nodes[-1].op_type == "Not"
    n3 = PaddleNode("n", "not_equal", inputs={"X": ["a"], "Y": ["b"]})
    outs = REDUCTION_OPS_MAPPING["not_equal"](builder, n3)
    assert builder.graph.nodes[-1].op_type == "Not"
    assert builder.graph.nodes[-2].op_type == "Equal"
    for op in [
        "logical_or",
        "logical_xor",
        "equal",
        "less_than",
        "less_equal",
        "greater_than",
        "greater_equal",
        "allclose",
    ]:
        n = PaddleNode("n", op, inputs={"X": ["a"], "Y": ["b"]})
        outs = REDUCTION_OPS_MAPPING[op](builder, n)
        assert len(outs) > 0


def test_reduction_ops_cumprod() -> None:
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "cumprod", inputs={"X": ["a"]})
    REDUCTION_OPS_MAPPING["cumprod"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Custom_Paddle_cumprod"
