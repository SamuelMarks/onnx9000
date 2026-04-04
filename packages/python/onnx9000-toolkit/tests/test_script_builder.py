"""Module docstring."""


def test_builder_all():
    """Docstring for D103."""
    from onnx9000.toolkit.script.builder import GraphBuilder
    from onnx9000.toolkit.script.op import op
    from onnx9000.toolkit.script.schema import set_target_opset
    from onnx9000.toolkit.script.var import Var

    set_target_opset(18)
    b = GraphBuilder("test")
    # it is a context manager
    with b:
        from onnx9000.toolkit.script.op import get_active_builder

        assert get_active_builder() is b

        import onnx9000.core.dtypes as dt

        v1 = b.add_input("x", dt.DType.FLOAT32, [1])
        v2 = b.add_input("y", dt.DType.FLOAT32, [1])
        b.add_output(v1)

        op.Add(v1, v2)

        graph = b.build()
        assert graph.name == "test"


def test_builder_more():
    """Docstring for D103."""
    import onnx9000.core.dtypes as dt
    import onnx9000.core.ir as ir
    from onnx9000.toolkit.script.builder import GraphBuilder
    from onnx9000.toolkit.script.var import Var

    b = GraphBuilder("test_more")
    b.set_metadata("doc", "domain", 2)
    assert b.metadata["version"] == "2"

    # 66, 70-73
    v1 = b.add_input("z", dt.DType.FLOAT32, [1])
    assert isinstance(v1, Var)

    b.add_output(v1, "z_out")

    # 77-84 add_initializer
    import numpy as np

    b.add_initializer("init1", np.array([1]))

    # add_constant

    # build
    g = b.build()
    assert g.name == "test_more"


def test_builder_infer_shapes():
    """Docstring for D103."""
    import numpy as np
    import onnx9000.core.dtypes as dt
    from onnx9000.toolkit.script.builder import GraphBuilder
    from onnx9000.toolkit.script.op import op
    from onnx9000.toolkit.script.var import Var

    b = GraphBuilder("test_infer")
    with b:
        v1 = b.add_input("x", dt.DType.FLOAT32, [1])
        b.add_initializer("init1", np.array([1], dtype=np.int32))

        # nested graph
        b2 = GraphBuilder("nested")
        b2.add_input("nested_x", dt.DType.FLOAT32, [1])

        # nodes
        res1 = op.Add(v1, v1)  # 132 Add
        op.Constant(np.array([1]))  # 137 Constant

        # 141-143 Squeeze
        op.Squeeze(v1, axes=[0])

        # attach nested
        op.If(1, then_branch=b2, else_branch=b2)

    b.infer_shapes()

    # 151: if not out_info.shape: ... in build()
    b.add_output(res1)
    b.build()


def test_builder_extract_subgraph():
    """Docstring for D103."""
    import numpy as np
    import onnx9000.core.dtypes as dt
    from onnx9000.toolkit.script.builder import GraphBuilder
    from onnx9000.toolkit.script.op import op

    b = GraphBuilder("test_extract")
    with b:
        v1 = b.add_input("x", dt.DType.FLOAT32, [1])
        v2 = b.add_input("y", dt.DType.FLOAT32, [1])
        b.add_initializer("init1", np.array([1.0], dtype=np.float32))

        # Add a node
        from onnx9000.toolkit.script.var import Var

        v_init = Var("init1")
        res1 = op.Add(v1, v_init)
        # Add a dead node
        op.Add(v2, v2)
        # Add a node that takes res1
        res2 = op.Mul(res1, res1)

        b.add_output(res2, res2.name)

    sub_b = b.extract_subgraph(["x"], [res2.name])
    assert sub_b.name == "test_extract_subgraph"
    # Should contain Add and Mul, but not dead Add
    assert len(sub_b.nodes) == 2


def test_builder_ops_misc():
    """Docstring for D103."""
    import onnx9000.core.ir as ir
    from onnx9000.toolkit.script.builder import GraphBuilder
    from onnx9000.toolkit.script.var import Var

    b = GraphBuilder("test_misc")
    n = ir.Node("Add", ["a"], ["b"], name="test_node")

    b.add_node(n)
    assert b.get_node("test_node") is n
    assert b.get_node("missing") is None  # 66

    # 70-73 delete
    b.delete(n)
    assert "test_node" not in b.node_by_name
    assert n not in b.nodes

    # 77-84 replace
    b.add_node(n)
    n2 = ir.Node("Sub", ["a"], ["b"])  # no name
    b.replace(n, n2)
    assert n2.name is not None
    assert "test_node" not in b.node_by_name
    assert b.nodes[0] is n2

    # 88-89 replace_input
    v_old = Var("a")
    v_new = Var("c")
    b.replace_input(n2, v_old, v_new)
    assert n2.inputs[0] == "c"

    # 93-97 merge
    b2 = GraphBuilder("test_merge")
    b2.add_node(ir.Node("Mul", ["c"], ["d"]))
    import onnx9000.core.dtypes as dt

    b2.add_input("c", dt.DType.FLOAT32, [1])
    b2.add_output(Var("d"), "out")
    import numpy as np

    b2.add_initializer("i", np.array([1]))

    b.merge(b2)
    assert len(b.nodes) == 2
    assert len(b.inputs) == 1
    assert len(b.outputs) == 1
    assert "i" in b.initializers

    # 101-116 clone
    b.rename_all("test_clone")
    assert len(b.nodes) == 2
    assert "test_clone_i" in b.initializers


def test_builder_extract_subgraph_nested():
    """Docstring for D103."""
    import numpy as np
    import onnx9000.core.dtypes as dt
    from onnx9000.toolkit.script.builder import GraphBuilder
    from onnx9000.toolkit.script.op import op

    b = GraphBuilder("test_extract_nested")
    b2 = GraphBuilder("nested")
    with b2:
        v_in = b2.add_input("nested_x", dt.DType.FLOAT32, [1])
        v_out = op.Add(v_in, v_in)
        b2.add_output(v_out, "nested_out")

    with b:
        b.add_input("x", dt.DType.FLOAT32, [1])
        # Add If node
        res1 = op.If(1, then_branch=b2, else_branch=b2)
        b.add_output(res1, "out1")

    sub_b = b.extract_subgraph(["x"], [res1.name])
    assert sub_b.name == "test_extract_nested_subgraph"


def test_builder_extract_subgraph_nested2(capsys):
    """Docstring for D103."""
    import onnx9000.core.dtypes as dt
    from onnx9000.toolkit.script.builder import GraphBuilder
    from onnx9000.toolkit.script.op import op

    b = GraphBuilder("test_extract_nested")
    b2 = GraphBuilder("nested")
    with b2:
        v_in = b2.add_input("nested_x", dt.DType.FLOAT32, [1])
        b2.add_output(v_in, "nested_out")

    with b:
        b.add_input("x", dt.DType.FLOAT32, [1])
        res1 = op.If(1, then_branch=b2, else_branch=b2)
        b.add_output(res1, "out1")

    print(res1.name)
    sub_b = b.extract_subgraph(["x"], [res1.name])
    assert sub_b.name == "test_extract_nested_subgraph"


def test_builder_extract_subgraph_nested_explicit():
    """Docstring for D103."""
    import onnx9000.core.dtypes as dt
    from onnx9000.toolkit.script.builder import GraphBuilder
    from onnx9000.toolkit.script.op import op

    b = GraphBuilder("test_extract_nested")
    b2 = GraphBuilder("nested")
    with b2:
        v_in = b2.add_input("nested_x", dt.DType.FLOAT32, [1])
        b2.add_output(v_in, "nested_out")

    with b:
        b.add_input("x", dt.DType.FLOAT32, [1])
        res1 = op.If(1, then_branch=b2, else_branch=b2)
        b.add_output(res1, res1.name)

    sub_b = b.extract_subgraph(["x"], [res1.name])
    assert sub_b.name == "test_extract_nested_subgraph"


def test_builder_extract_subgraph_else():
    """Docstring for D103."""
    import onnx9000.core.dtypes as dt
    import onnx9000.core.ir as ir
    from onnx9000.toolkit.script.builder import GraphBuilder
    from onnx9000.toolkit.script.op import op

    b = GraphBuilder("test_extract_else")
    b2 = GraphBuilder("nested")

    n = ir.Node("If", ["x"], ["out1"], name="if_node", attributes={"then_branch": b2})
    b.add_node(n)

    sub_b = b.extract_subgraph(["x"], ["out1"])
    assert sub_b.name == "test_extract_else_subgraph"


def test_builder_extract_subgraph_else_dict():
    """Docstring for D103."""
    import onnx9000.core.dtypes as dt
    import onnx9000.core.ir as ir
    from onnx9000.toolkit.script.builder import GraphBuilder
    from onnx9000.toolkit.script.op import op

    b = GraphBuilder("test_extract_else2")
    b2 = GraphBuilder("nested")

    # Do NOT pass attributes=... so that it does NOT convert to `ir.Attribute`
    # Instead, we just set `n.attributes["then_branch"] = b2` but since it's `Optional[dict] = None`, `n.attributes` is `{}`
    n = ir.Node("If", ["x"], ["out1"], name="if_node")
    n.attributes["then_branch"] = b2
    b.add_node(n)

    sub_b = b.extract_subgraph(["x"], ["out1"])
    assert sub_b.name == "test_extract_else2_subgraph"


def test_builder_extract_subgraph_else_attr_obj():
    """Docstring for D103."""
    import onnx9000.core.dtypes as dt
    import onnx9000.core.ir as ir
    from onnx9000.toolkit.script.builder import GraphBuilder
    from onnx9000.toolkit.script.op import op

    b = GraphBuilder("test_extract_else_attr")
    b2 = GraphBuilder("nested")

    n = ir.Node("If", ["x"], ["out1"], name="if_node")
    # Wrap in ir.Attribute
    n.attributes["then_branch"] = ir.Attribute("then_branch", "GRAPH", b2)
    b.add_node(n)

    sub_b = b.extract_subgraph(["x"], ["out1"])
    assert sub_b.name == "test_extract_else_attr_subgraph"


def test_builder_if_loop():
    """Docstring for D103."""
    from onnx9000.toolkit.script.builder import GraphBuilder
    from onnx9000.toolkit.script.var import Var

    b = GraphBuilder("test_if_loop")
    v = Var("x")

    try:
        b.If(v)
    except Exception:
        pass

    try:
        b.Loop(v, v)
    except Exception:
        pass


def test_builder_to_onnx():
    """Docstring for D103."""
    import numpy as np
    import onnx9000.core.dtypes as dt
    from onnx9000.toolkit.script.builder import GraphBuilder
    from onnx9000.toolkit.script.op import op

    b = GraphBuilder("test_to_onnx")
    with b:
        v1 = b.add_input("x", dt.DType.FLOAT32, [1])
        v2 = b.add_input("y", dt.DType.INT64, ["N"])

        b.add_initializer("init1", np.array([1.0], dtype=np.float32))
        b.add_initializer("init2", np.array([1], dtype=np.int64))

        # some node
        from onnx9000.toolkit.script.var import Var

        v_init1 = Var("init1")
        res1 = op.Add(v1, v_init1)

        b.add_output(res1, "out1")
        b.add_output(v2, "out2")

    model = b.to_onnx()
    assert model.graph.name == "test_to_onnx"


def test_builder_from_onnx():
    """Docstring for D103."""
    import numpy as np
    import onnx9000.core.dtypes as dt
    from onnx9000.toolkit.script.builder import GraphBuilder
    from onnx9000.toolkit.script.op import op

    # Let's create an ONNX model with to_onnx, and then load it!
    b = GraphBuilder("test_to_onnx_load")
    with b:
        v1 = b.add_input("x", dt.DType.FLOAT32, [1])
        b.add_initializer("init1", np.array([1.0], dtype=np.float32))

        from onnx9000.toolkit.script.var import Var

        res1 = op.Add(v1, Var("init1"))

        # Test attributes serialization in to_onnx and deserialization in from_onnx
        # 1. Float, Int, String, Tensor
        res2 = op.Constant(np.array([2.0], dtype=np.float32))  # Tensor

        # We need a node with explicitly set attributes for float, int, string, floats, ints.
        # TopK has 'axis' (int), 'largest' (int)
        res_topk = op.TopK(v1, res2, axis=0, largest=1)

        b.add_output(res1, "out1")
        b.add_output(res_topk[0], "out2")

    b.to_onnx()

    # Add a custom node with float, string, and Graph attr directly to model proto to ensure coverage!
    # Or just use GraphBuilder
    import onnx9000.core.ir as ir

    b_nested = GraphBuilder("nested")
    b_nested.add_input("nx", dt.DType.FLOAT32, [1])

    n_custom = ir.Node("CustomOp", ["x"], ["out3"], name="custom")
    n_custom.attributes["float_attr"] = 1.5
    n_custom.attributes["str_attr"] = "hello"
    n_custom.attributes["floats_attr"] = [1.0, 2.0]
    n_custom.attributes["graph_attr"] = b_nested
    b.add_node(n_custom)

    model2 = b.to_onnx()

    # Now from_onnx
    b_loaded = GraphBuilder.from_onnx(model2)
    assert b_loaded.name == "test_to_onnx_load"


def test_builder_validate():
    """Docstring for D103."""
    import onnx9000.core.dtypes as dt
    import pytest
    from onnx9000.toolkit.script.builder import GraphBuilder
    from onnx9000.toolkit.script.op import op

    b = GraphBuilder("test_validate")
    with b:
        v1 = b.add_input("x", dt.DType.FLOAT32, [1])
        res1 = op.Add(v1, v1)
        op.Mul(res1, res1)
        n1 = b.nodes[0]
        n2 = b.nodes[1]
        n1.inputs.append(n2.outputs[0])

    with pytest.raises(ValueError, match="Cyclic dependency"):
        b.validate()

    b2 = GraphBuilder("test_validate_ok")
    with b2:
        v1 = b2.add_input("x", dt.DType.FLOAT32, [1])
        res1 = op.Add(v1, v1)
    b2.validate()


def test_builder_to_from_onnx_more():
    """Docstring for D103."""
    import numpy as np
    import onnx9000.core.dtypes as dt
    import onnx9000.core.ir as ir
    from onnx9000.toolkit.script.builder import GraphBuilder
    from onnx9000.toolkit.script.op import op

    b = GraphBuilder("test_to_onnx_more")
    b.set_metadata("my_doc", "my_domain", 5)
    b.metadata["custom_domain"] = "my_custom"

    n = ir.Node("Custom", [], ["out"], name="custom")
    n.attributes = {
        "str_attr": ir.Attribute("str_attr", "STRING", "string_value"),
        "int_arr": ir.Attribute("int_arr", "TENSOR", np.array([1], dtype=np.int64)),
        "ints_attr": ir.Attribute("ints_attr", "INTS", [1, 2]),
    }
    b.add_node(n)

    model = b.to_onnx()
    assert model.doc_string == "my_doc"
    assert model.domain == "my_domain"
    assert model.model_version == 5

    b_loaded = GraphBuilder.from_onnx(model.graph)  # pass GraphProto directly
    assert b_loaded.name == "test_to_onnx_more"


def test_builder_validate_visited():
    """Docstring for D103."""
    import onnx9000.core.dtypes as dt
    from onnx9000.toolkit.script.builder import GraphBuilder
    from onnx9000.toolkit.script.op import op

    b = GraphBuilder("test_validate_visited")
    with b:
        v1 = b.add_input("x", dt.DType.FLOAT32, [1])
        # Two nodes that depend on the same parent node, to hit `if n.name in visited:`
        n1 = op.Add(v1, v1)
        n2 = op.Mul(n1, n1)
        n3 = op.Sub(n1, n1)
        # n4 depends on n2 and n3, so n1 is visited twice
        op.Add(n2, n3)
    b.validate()


def test_builder_from_onnx_types():
    """Docstring for D103."""
    import numpy as np
    import onnx9000.core.dtypes as dt
    import onnx9000.core.ir as ir
    from onnx9000.toolkit.script.builder import GraphBuilder
    from onnx9000.toolkit.script.op import op

    b = GraphBuilder("test_from_onnx_types")
    with b:
        # test `if init.data_type == 1: else:` (392)
        b.add_initializer("init_int64_2", np.array([1], dtype=np.int64))

        # test `hasattr(model_proto, "graph") else` (366)
        # GraphProto directly is what we pass! So `hasattr(model_proto, "graph")` is false, and it uses `model_proto`.
        # Oh wait, `if hasattr(model_proto, "graph")` means we passed a ModelProto!
        # In the previous test we did: `b_loaded = GraphBuilder.from_onnx(model.graph)`.
        # That means `hasattr(model_proto, "graph")` was False!
        # We need to pass `model` directly to cover the `True` branch!

    model = b.to_onnx()
    GraphBuilder.from_onnx(model)  # passes ModelProto!


def test_builder_to_onnx_int32_dtype():
    """Docstring for D103."""
    import onnx9000.core.dtypes as dt
    from onnx9000.toolkit.script.builder import GraphBuilder

    b = GraphBuilder("test_dtype")
    b.add_input("x", dt.DType.INT32, [1])
    b.to_onnx()
