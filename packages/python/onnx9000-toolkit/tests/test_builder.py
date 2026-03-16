import numpy as np
import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Node
from onnx9000.toolkit.script.builder import GraphBuilder
from onnx9000.toolkit.script.var import Var


def test_graph_builder_basic() -> None:
    builder = GraphBuilder(name="TestGraph")
    assert builder.name == "TestGraph"
    builder.add_input("x", DType.FLOAT32, (1, 3, 224, 224))
    builder.add_input("y", DType.FLOAT32, (1, 3, 224, 224))
    assert len(builder.inputs) == 2
    node = Node(op_type="Add", inputs=["x", "y"], outputs=["z"], attributes={})
    builder.add_node(node)
    assert len(builder.nodes) == 1
    builder.add_output(Var(name="z"))
    assert len(builder.outputs) == 1
    graph = builder.build()
    assert graph.name == "TestGraph"
    assert len(graph.nodes) == 1
    assert graph.inputs == ["x", "y"]
    assert graph.outputs == ["z"]


def test_graph_builder_initializer() -> None:
    builder = GraphBuilder()
    arr = np.array([1, 2, 3], dtype=np.float32)
    builder.add_initializer("init1", arr)
    assert "init1" in builder.initializers
    graph = builder.build()
    assert "init1" in graph.initializers
    assert "init1" in graph.tensors


def test_graph_builder_metadata() -> None:
    builder = GraphBuilder()
    builder.set_metadata(doc_string="doc", domain="my.domain", version=2)
    model = builder.to_onnx()
    assert model.doc_string == "doc"
    assert model.domain == "my.domain"
    assert model.model_version == 2


def test_graph_builder_to_onnx() -> None:
    builder = GraphBuilder()
    builder.add_input("x", DType.FLOAT32, (1, 10))
    node = Node(
        op_type="Relu", inputs=["x"], outputs=["y"], attributes={"alpha": 0.1, "ints": [1, 2]}
    )
    builder.add_node(node)
    builder.add_output(Var(name="y"))
    model = builder.to_onnx()
    assert model.graph.name == "Graph"
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "Relu"


def test_graph_builder_from_onnx() -> None:
    builder = GraphBuilder()
    builder.add_input("x", DType.FLOAT32, (1, 10))
    node = Node(
        op_type="Relu", inputs=["x"], outputs=["y"], attributes={"alpha": 0.1, "ints": [1, 2]}
    )
    builder.add_node(node)
    builder.add_output(Var(name="y"))
    builder.add_initializer("w", np.array([1.0], dtype=np.float32))
    model = builder.to_onnx()
    builder2 = GraphBuilder.from_onnx(model)
    assert builder2.name == "Graph"
    assert len(builder2.inputs) == 1
    assert len(builder2.nodes) == 1
    assert builder2.nodes[0].op_type == "Relu"
    assert builder2.nodes[0].attributes["alpha"] == pytest.approx(0.1)
    assert builder2.nodes[0].attributes["ints"] == [1, 2]


def test_graph_builder_validate() -> None:
    builder = GraphBuilder()
    node1 = Node(op_type="Relu", inputs=["a"], outputs=["b"], attributes={})
    node2 = Node(op_type="Relu", inputs=["b"], outputs=["a"], attributes={})
    builder.add_node(node1)
    builder.add_node(node2)
    with pytest.raises(ValueError, match="Cyclic dependency detected"):
        builder.validate()


def test_graph_builder_edit_apis() -> None:
    builder = GraphBuilder()
    node1 = Node(op_type="Relu", inputs=["a"], outputs=["b"], attributes={})
    node2 = Node(op_type="Add", inputs=["b", "c"], outputs=["d"], attributes={})
    builder.add_node(node1)
    builder.add_node(node2)
    assert builder.get_node(node1.name) == node1
    node3 = Node(op_type="Sigmoid", inputs=["a"], outputs=["b"], attributes={})
    builder.replace(node1, node3)
    assert node3 in builder.nodes
    assert node1 not in builder.nodes
    builder.delete(node2)
    assert node2 not in builder.nodes
    node4 = Node(op_type="Log", inputs=["b"], outputs=["e"], attributes={})
    builder.add_node(node4)
    v_new = Var(name="new_b")
    v_old = Var(name="b")
    builder.replace_input(node4, v_old, v_new)
    assert node4.inputs == ["new_b"]


def test_graph_builder_merge_and_rename() -> None:
    b1 = GraphBuilder("G1")
    b1.add_input("x", DType.FLOAT32, (10,))
    b1.add_node(Node("Relu", ["x"], ["y"], {}))
    b2 = GraphBuilder("G2")
    b2.add_input("y", DType.FLOAT32, (10,))
    b2.add_node(Node("Log", ["y"], ["z"], {}))
    b2.rename_all("sub")
    assert b2.inputs[0]["name"] == "sub_y"
    b1.merge(b2)
    assert len(b1.nodes) == 2
    assert len(b1.inputs) == 2


def test_graph_builder_all_attributes() -> None:
    builder = GraphBuilder()
    attrs = {
        "a_float": 1.5,
        "an_int": 42,
        "a_string": "hello",
        "a_tensor": np.array([1, 2], dtype=np.int64),
        "a_tensor_f": np.array([1.5], dtype=np.float32),
        "some_ints": [1, 2, 3],
        "some_floats": [1.1, 2.2],
    }
    node = Node(op_type="Dummy", inputs=[], outputs=[], attributes=attrs)
    builder.add_node(node)
    model = builder.to_onnx()
    builder2 = GraphBuilder.from_onnx(model)
    restored_attrs = builder2.nodes[0].attributes
    assert restored_attrs["a_float"] == pytest.approx(1.5)
    assert restored_attrs["an_int"] == 42
    assert restored_attrs["a_string"] == "hello"
    assert restored_attrs["some_ints"] == [1, 2, 3]
    assert restored_attrs["some_floats"] == pytest.approx([1.1, 2.2])


def test_graph_builder_extract_subgraph() -> None:
    builder = GraphBuilder()
    builder.add_input("a", DType.FLOAT32, (1,))
    builder.add_node(Node("Relu", ["a"], ["b"], {}))
    builder.add_node(Node("Log", ["b"], ["c"], {}))
    builder.add_node(Node("Exp", ["b"], ["d"], {}))
    builder.add_output(Var("c"))
    builder.add_output(Var("d"))
    sub = builder.extract_subgraph(["a"], ["c"])
    assert len(sub.nodes) == 2


def test_graph_builder_control_flow() -> None:
    builder = GraphBuilder()
    a = builder.add_input("a", DType.FLOAT32, (1,))
    cond = builder.add_input("cond", DType.BOOL, (1,))
    if_ctx = builder.If(cond, num_outputs=1)
    with if_ctx.Then():
        from onnx9000.toolkit.script import op

        t_out = op.Relu(a)
        if_ctx.then_builder.add_output(t_out)
    with if_ctx.Else():
        from onnx9000.toolkit.script import op

        e_out = op.Sigmoid(a)
        if_ctx.else_builder.add_output(e_out)
    if_ctx.build()
    assert len(builder.nodes) == 1
    assert builder.nodes[0].op_type == "If"
    assert "then_branch" in builder.nodes[0].attributes
    mtc = builder.add_input("mtc", DType.INT64, (1,))
    loop_ctx = builder.Loop(mtc, cond, num_outputs=1)
    with loop_ctx.Body():
        from onnx9000.toolkit.script import op

        b_out = op.Relu(a)
        loop_ctx.body_builder.add_output(b_out)
    loop_ctx.build()
    assert len(builder.nodes) == 2
    assert builder.nodes[1].op_type == "Loop"
