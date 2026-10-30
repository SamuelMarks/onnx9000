"""Tests the math ops module functionality."""

from onnx9000.converters.paddle.builder import PaddleToONNXGraphBuilder
from onnx9000.converters.paddle.math_ops import MATH_OPS_MAPPING
from onnx9000.converters.paddle.parsers import PaddleNode


def test_paddle_math_ops_simple() -> None:
    """Tests the paddle math ops simple functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "elementwise_add", inputs={"X": ["a"], "Y": ["b"]})
    MATH_OPS_MAPPING["elementwise_add"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Add"
    n = PaddleNode("n", "abs", inputs={"X": ["a"]})
    MATH_OPS_MAPPING["abs"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Abs"


def test_paddle_math_ops_floordiv() -> None:
    """Tests the paddle math ops floordiv functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "elementwise_floordiv", inputs={"X": ["a"], "Y": ["b"]})
    MATH_OPS_MAPPING["elementwise_floordiv"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Floor"
    assert builder.graph.nodes[-2].op_type == "Div"


def test_paddle_math_ops_log1p() -> None:
    """Tests the paddle math ops log1p functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "log1p", inputs={"X": ["a"]})
    MATH_OPS_MAPPING["log1p"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Log"
    assert builder.graph.nodes[-2].op_type == "Add"


def test_paddle_math_ops_rsqrt() -> None:
    """Tests the paddle math ops rsqrt functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "rsqrt", inputs={"X": ["a"]})
    MATH_OPS_MAPPING["rsqrt"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Reciprocal"
    assert builder.graph.nodes[-2].op_type == "Sqrt"


def test_paddle_math_ops_square() -> None:
    """Tests the paddle math ops square functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "square", inputs={"X": ["a"]})
    MATH_OPS_MAPPING["square"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Mul"
    assert builder.graph.nodes[-1].inputs == ["a", "a"]


def test_paddle_math_ops_isfinite() -> None:
    """Tests the paddle math ops isfinite functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "isfinite", inputs={"X": ["a"]})
    MATH_OPS_MAPPING["isfinite"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Not"
    assert builder.graph.nodes[-2].op_type == "Or"
    assert builder.graph.nodes[-3].op_type == "IsInf"
    assert builder.graph.nodes[-4].op_type == "IsNaN"


def test_paddle_math_ops_scale() -> None:
    """Tests the paddle math ops scale functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "scale", inputs={"X": ["a"]}, attrs={"scale": 2.0, "bias": 1.0})
    MATH_OPS_MAPPING["scale"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Add"
    assert builder.graph.nodes[-2].op_type == "Mul"


def test_paddle_math_ops_sum() -> None:
    """Tests the paddle math ops sum functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "sum", inputs={"X": ["a", "b", "c"]})
    MATH_OPS_MAPPING["sum"](builder, n)
    assert builder.graph.nodes[-1].op_type == "Sum"


def test_paddle_math_ops_custom() -> None:
    """Tests the paddle math ops custom functionality."""
    builder = PaddleToONNXGraphBuilder()
    for op in ["log2", "log10", "clip"]:
        n = PaddleNode("n", op, inputs={"X": ["a"]})
        MATH_OPS_MAPPING[op](builder, n)
        assert builder.graph.nodes[-1].op_type in ["Div", "Clip"]


def test_paddle_math_ops_custom_y() -> None:
    """Tests the paddle math ops custom y functionality."""
    from onnx9000.converters.paddle.builder import PaddleToONNXGraphBuilder
    from onnx9000.converters.paddle.math_ops import _map_custom
    from onnx9000.converters.paddle.parsers import PaddleNode

    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "Custom_Log2", inputs={"X": ["a"], "Y": ["b"]})
    _map_custom("Custom_Log2")(builder, n)
    assert builder.graph.nodes[-1].op_type == "Custom_Log2"
    assert builder.graph.nodes[-1].inputs == ["a", "b"]


def test_map_dot() -> None:
    """Tests the map dot functionality."""
    builder = PaddleToONNXGraphBuilder("test")
    node = PaddleNode("dot", {"X": ["x"], "Y": ["y"]}, {"out": ["z"]}, {}, "dot_node")
    MATH_OPS_MAPPING["dot"](builder, node)
    assert len(builder.graph.nodes) == 2
    assert builder.graph.nodes[0].op_type == "Mul"
    assert builder.graph.nodes[1].op_type == "ReduceSum"


def test_map_cross() -> None:
    """Tests the map cross functionality."""
    builder = PaddleToONNXGraphBuilder("test")
    node = PaddleNode("cross", {"X": ["x"], "Y": ["y"]}, {"out": ["z"]}, {"dim": 1}, "cross_node")
    MATH_OPS_MAPPING["cross"](builder, node)
    assert len(builder.graph.nodes) == 1
    assert builder.graph.nodes[0].op_type == "Custom_Paddle_cross"
