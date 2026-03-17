"""Tests the builder module functionality."""

from onnx9000.converters.paddle.builder import PaddleToONNXGraphBuilder
from onnx9000.converters.paddle.parsers import PaddleNode
from onnx9000.core.ir import Node


def test_paddle_builder_unique_name() -> None:
    """Tests the paddle builder unique name functionality."""
    builder = PaddleToONNXGraphBuilder()
    assert builder.get_unique_name("test") == "test"
    assert builder.get_unique_name("test") == "test_1"


def test_paddle_builder_add_constant() -> None:
    """Tests the paddle builder add constant functionality."""
    builder = PaddleToONNXGraphBuilder()
    name = builder.add_constant("c", 42.0, 1, (1,))
    assert name == "c"
    assert "c" in builder.graph.initializers
    assert builder.graph.tensors["c"].shape == (1,)


def test_paddle_builder_infer_shape() -> None:
    """Tests the paddle builder infer shape functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "matmul_v2")
    assert builder.infer_shape(n, [(2, 3), (3, 4)]) == (2, 4)
    assert builder.infer_shape(n, [(), ()]) == ()
    assert builder.infer_shape(PaddleNode("n", "other"), [(2, 3)]) == ()


def test_paddle_builder_extract_attr() -> None:
    """Tests the paddle builder extract attr functionality."""
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "op", attrs={"a": 1, "b": [1, 2]})
    assert builder.extract_attr(n, "a") == 1
    assert builder.extract_attr(n, "missing", 5) == 5
    assert builder.extract_list_attr(n, "b") == [1, 2]
    assert builder.extract_list_attr(n, "a") == [1]
    assert builder.extract_list_attr(n, "missing") == []


def test_paddle_builder_broadcasting() -> None:
    """Tests the paddle builder broadcasting functionality."""
    builder = PaddleToONNXGraphBuilder()
    assert builder.resolve_broadcasting((1, 2, 3), (2, 3)) == (1, 2, 3)


def test_paddle_builder_flatten_lod() -> None:
    """Tests the paddle builder flatten lod functionality."""
    builder = PaddleToONNXGraphBuilder()
    outs = builder.flatten_lod_tensor("in", "out")
    assert len(outs) == 1
    assert outs[0] == "out"
    assert builder.graph.nodes[-1].op_type == "Flatten"


def test_paddle_builder_make_node() -> None:
    """Tests the paddle builder make node functionality."""
    builder = PaddleToONNXGraphBuilder()
    outs = builder.make_node("Relu", ["in"], {}, "relu")
    assert len(outs) == 1
    assert outs[0] == "relu_out_0"
    assert builder.graph.nodes[-1].op_type == "Relu"
    outs = builder.make_node("Split", ["in"], {}, "split", outputs=["o1", "o2"])
    assert len(outs) == 2
    assert outs == ["o1", "o2"]


def test_paddle_builder_make_node_optional() -> None:
    """Tests the paddle builder make node optional functionality."""
    builder = PaddleToONNXGraphBuilder()
    builder.make_node_optional_inputs("Concat", ["in", None], {}, "concat")
    assert builder.graph.nodes[-1].inputs == ["in", ""]


def test_paddle_builder_replace() -> None:
    """Tests the paddle builder replace functionality."""
    builder = PaddleToONNXGraphBuilder()
    builder.make_node("Relu", ["in"], {}, "relu")
    old_name = builder.graph.nodes[-1].name
    new_node = Node(
        op_type="Sigmoid", inputs=["in"], outputs=["relu_out_0"], name="sig", attributes={}
    )
    builder.replace_node(old_name, new_node)
    assert builder.graph.nodes[0].op_type == "Sigmoid"


def test_paddle_builder_misc() -> None:
    """Tests the paddle builder misc functionality."""
    builder = PaddleToONNXGraphBuilder()
    assert builder.resolve_variable("v") is None
    assert "nodes." in builder.dump_ir()
