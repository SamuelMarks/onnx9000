from onnx9000.core.ir import Node
from onnx9000.converters.paddle.builder import PaddleToONNXGraphBuilder
from onnx9000.converters.paddle.parsers import PaddleNode


def test_paddle_builder_unique_name() -> None:
    builder = PaddleToONNXGraphBuilder()
    assert builder.get_unique_name("test") == "test"
    assert builder.get_unique_name("test") == "test_1"


def test_paddle_builder_add_constant() -> None:
    builder = PaddleToONNXGraphBuilder()
    name = builder.add_constant("c", 42.0, 1, (1,))
    assert name == "c"
    assert "c" in builder.graph.initializers
    assert builder.graph.tensors["c"].shape == (1,)


def test_paddle_builder_infer_shape() -> None:
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "matmul_v2")
    assert builder.infer_shape(n, [(2, 3), (3, 4)]) == (2, 4)
    assert builder.infer_shape(n, [(), ()]) == ()
    assert builder.infer_shape(PaddleNode("n", "other"), [(2, 3)]) == ()


def test_paddle_builder_extract_attr() -> None:
    builder = PaddleToONNXGraphBuilder()
    n = PaddleNode("n", "op", attrs={"a": 1, "b": [1, 2]})
    assert builder.extract_attr(n, "a") == 1
    assert builder.extract_attr(n, "missing", 5) == 5
    assert builder.extract_list_attr(n, "b") == [1, 2]
    assert builder.extract_list_attr(n, "a") == [1]
    assert builder.extract_list_attr(n, "missing") == []


def test_paddle_builder_broadcasting() -> None:
    builder = PaddleToONNXGraphBuilder()
    assert builder.resolve_broadcasting((1, 2, 3), (2, 3)) == (1, 2, 3)


def test_paddle_builder_flatten_lod() -> None:
    builder = PaddleToONNXGraphBuilder()
    outs = builder.flatten_lod_tensor("in", "out")
    assert len(outs) == 1
    assert outs[0] == "out"
    assert builder.graph.nodes[-1].op_type == "Flatten"


def test_paddle_builder_make_node() -> None:
    builder = PaddleToONNXGraphBuilder()
    outs = builder.make_node("Relu", ["in"], {}, "relu")
    assert len(outs) == 1
    assert outs[0] == "relu_out_0"
    assert builder.graph.nodes[-1].op_type == "Relu"
    outs = builder.make_node("Split", ["in"], {}, "split", outputs=["o1", "o2"])
    assert len(outs) == 2
    assert outs == ["o1", "o2"]


def test_paddle_builder_make_node_optional() -> None:
    builder = PaddleToONNXGraphBuilder()
    builder.make_node_optional_inputs("Concat", ["in", None], {}, "concat")
    assert builder.graph.nodes[-1].inputs == ["in", ""]


def test_paddle_builder_replace() -> None:
    builder = PaddleToONNXGraphBuilder()
    builder.make_node("Relu", ["in"], {}, "relu")
    old_name = builder.graph.nodes[-1].name
    new_node = Node(
        op_type="Sigmoid", inputs=["in"], outputs=["relu_out_0"], name="sig", attributes={}
    )
    builder.replace_node(old_name, new_node)
    assert builder.graph.nodes[0].op_type == "Sigmoid"


def test_paddle_builder_misc() -> None:
    builder = PaddleToONNXGraphBuilder()
    assert builder.resolve_variable("v") is None
    assert "nodes." in builder.dump_ir()
