import pytest
from onnx9000.frontends.paddle.builder import PaddleToONNXGraphBuilder
from onnx9000.frontends.paddle.parsers import PaddleNode
from onnx9000.frontends.paddle.math_ops import MATH_OPS_MAPPING


def test_paddle_math_ops_simple():
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "elementwise_add", inputs={"X": ["a"], "Y": ["b"]})
    outs = MATH_OPS_MAPPING["elementwise_add"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Add"
    n = PaddleNode("n", "abs", inputs={"X": ["a"]})
    outs = MATH_OPS_MAPPING["abs"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Abs"


def test_paddle_math_ops_floordiv():
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "elementwise_floordiv", inputs={"X": ["a"], "Y": ["b"]})
    outs = MATH_OPS_MAPPING["elementwise_floordiv"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Floor"
    assert builder.graph.nodes[-2].op_type == "Div"


def test_paddle_math_ops_log1p():
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "log1p", inputs={"X": ["a"]})
    outs = MATH_OPS_MAPPING["log1p"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Log"
    assert builder.graph.nodes[-2].op_type == "Add"


def test_paddle_math_ops_rsqrt():
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "rsqrt", inputs={"X": ["a"]})
    outs = MATH_OPS_MAPPING["rsqrt"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Reciprocal"
    assert builder.graph.nodes[-2].op_type == "Sqrt"


def test_paddle_math_ops_square():
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "square", inputs={"X": ["a"]})
    outs = MATH_OPS_MAPPING["square"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Mul"
    assert builder.graph.nodes[-1].inputs == ["a", "a"]


def test_paddle_math_ops_isfinite():
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "isfinite", inputs={"X": ["a"]})
    outs = MATH_OPS_MAPPING["isfinite"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Not"
    assert builder.graph.nodes[-2].op_type == "Or"
    assert builder.graph.nodes[-3].op_type == "IsInf"
    assert builder.graph.nodes[-4].op_type == "IsNaN"


def test_paddle_math_ops_scale():
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "scale", inputs={"X": ["a"]}, attrs={"scale": 2.0, "bias": 1.0})
    outs = MATH_OPS_MAPPING["scale"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Add"
    assert builder.graph.nodes[-2].op_type == "Mul"


def test_paddle_math_ops_sum():
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "sum", inputs={"X": ["a", "b", "c"]})
    outs = MATH_OPS_MAPPING["sum"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Sum"


def test_paddle_math_ops_custom():
    builder = PaddleToONNXGraphBuilder()
    for op in ["log2", "log10", "clip"]:
        n = PaddleNode("n", op, inputs={"X": ["a"]})
        outs = MATH_OPS_MAPPING[op](builder, n)
        assert builder.graph.nodes[-1].op_type in ["Div", "Clip"]


def test_paddle_math_ops_custom_y():
    from onnx9000.frontends.paddle.builder import PaddleToONNXGraphBuilder
    from onnx9000.frontends.paddle.parsers import PaddleNode
    from onnx9000.frontends.paddle.math_ops import _map_custom

    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "Custom_Log2", inputs={"X": ["a"], "Y": ["b"]})
    outs = _map_custom("Custom_Log2")(builder, n)
    assert builder.graph.nodes[-1].op_type == "Custom_Log2"
    assert builder.graph.nodes[-1].inputs == ["a", "b"]
