import pytest
from onnx9000.frontends.tf.builder import TFToONNXGraphBuilder
from onnx9000.frontends.tf.parsers import TFNode
from onnx9000.core.ir import Node


def test_builder_get_unique_name():
    builder = TFToONNXGraphBuilder()
    assert builder.get_unique_name("node") == "node"
    assert builder.get_unique_name("node") == "node_1"
    assert builder.get_unique_name("node") == "node_2"


def test_builder_add_constant():
    builder = TFToONNXGraphBuilder()
    name = builder.add_constant("const_1", 42, 1, (1,))
    assert name == "const_1"
    assert "const_1" in builder.graph.initializers
    assert builder.graph.tensors["const_1"].shape == (1,)


def test_builder_infer_shape():
    builder = TFToONNXGraphBuilder()
    node = TFNode("mat1", "MatMul")
    shape = builder.infer_shape(node, [(2, 3), (3, 4)])
    assert shape == (2, 4)
    shape = builder.infer_shape(node, [(), ()])
    assert shape == ()
    shape = builder.infer_shape(TFNode("add1", "Add"), [(2, 3)])
    assert shape == ()


def test_builder_infer_dtype():
    builder = TFToONNXGraphBuilder()
    assert builder.infer_dtype(TFNode("n", "Add"), [3, 3]) == 3
    assert builder.infer_dtype(TFNode("n", "Add"), []) == 1


def test_builder_convert_nhwc_to_nchw():
    builder = TFToONNXGraphBuilder()
    out = builder.convert_nhwc_to_nchw("input_tensor")
    assert out.startswith("transpose_nchw")
    assert len(builder.graph.nodes) == 1
    assert builder.graph.nodes[0].op_type == "Transpose"
    assert builder.graph.nodes[0].inputs == ["input_tensor"]


def test_builder_calc_dynamic_padding():
    builder = TFToONNXGraphBuilder()
    assert builder.calc_dynamic_padding("VALID", (1, 5, 5, 1), (3, 3), [1, 1]) == [
        0,
        0,
        0,
        0,
    ]
    assert builder.calc_dynamic_padding("SAME", (1, 5, 5, 1), (3, 3), [1, 1]) == [
        1,
        1,
        1,
        1,
    ]


def test_builder_extract_attr():
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "op", attr={"val": 10})
    assert builder.extract_attr(node, "val") == 10
    assert builder.extract_attr(node, "missing", 5) == 5


def test_builder_resolve_broadcasting():
    builder = TFToONNXGraphBuilder()
    assert builder.resolve_broadcasting((2, 3), (3,)) == (2, 3)
    assert builder.resolve_broadcasting((3,), (2, 3)) == (2, 3)


def test_builder_make_node():
    builder = TFToONNXGraphBuilder()
    outs = builder.make_node("Relu", ["in_1"], {"attr1": 1}, "relu_node")
    assert len(outs) == 1
    assert builder.graph.nodes[-1].op_type == "Relu"
    assert builder.graph.nodes[-1].attributes == {"attr1": 1}


def test_builder_make_node_optional_inputs():
    builder = TFToONNXGraphBuilder()
    outs = builder.make_node_optional_inputs(
        "Concat", ["in_1", None, "in_2"], {}, "concat_node"
    )
    assert builder.graph.nodes[-1].inputs == ["in_1", "", "in_2"]


def test_builder_replace_node():
    builder = TFToONNXGraphBuilder()
    outs = builder.make_node("Relu", ["in_1"], {}, "relu_node")
    old_node_name = builder.graph.nodes[0].name
    new_node = Node(
        op_type="Sigmoid",
        inputs=["in_1"],
        outputs=outs,
        name="sigmoid_node",
        attributes={},
    )
    builder.replace_node(old_node_name, new_node)
    assert builder.graph.nodes[0].op_type == "Sigmoid"
    builder.replace_node("non_existent", new_node)


def test_builder_rewire_edge():
    builder = TFToONNXGraphBuilder()
    outs1 = builder.make_node("Relu", ["in_1"], {}, "relu_1")
    outs2 = builder.make_node("Relu", [outs1[0]], {}, "relu_2")
    builder.rewire_edge(outs1[0], "new_input")
    assert builder.graph.nodes[1].inputs == ["new_input"]


def test_builder_extract_const_value():
    builder = TFToONNXGraphBuilder()
    node = TFNode("n", "Const", attr={"value": 99})
    assert builder.extract_const_value(node) == 99


def test_builder_resolve_variable():
    builder = TFToONNXGraphBuilder()
    assert builder.resolve_variable("var_x") is None
