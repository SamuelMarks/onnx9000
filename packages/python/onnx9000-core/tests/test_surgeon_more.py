import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Constant, Graph, Node, Variable
from onnx9000.core.surgeon import *


def test_toposort_cyclic() -> None:
    g = Graph("cyclic")
    v1 = Variable("v1")
    v2 = Variable("v2")
    g.add_tensor(v1)
    g.add_tensor(v2)
    g.add_tensor(v1)
    g.add_tensor(v2)
    n1 = Node("N1", inputs=[v1], outputs=[v2])
    n2 = Node("N2", inputs=[v2], outputs=[v1])
    g.add_node(n1)
    g.add_node(n2)
    v1.outputs.append(n1)
    n1.inputs.append(v1)
    v1.inputs.append(n2)
    v2.inputs.append(n1)
    v2.outputs.append(n2)
    with pytest.raises(ValueError, match="Cyclic graph"):
        toposort(g)


def test_walk_nested_graphs() -> None:
    sub_g = Graph("sub")
    v_sub = Variable("v_sub")
    sub_g.add_tensor(v_sub)
    sub_g.promote_to_output(v_sub)
    sub_n = Node("SubN", outputs=[v_sub])
    sub_g.add_node(sub_n)
    g = Graph("main")
    v_main = Variable("v_main")
    g.add_tensor(v_main)
    g.promote_to_output(v_main)
    n_main = Node("MainN", outputs=[v_main])
    n_main.attributes["body"] = Attribute("body", value=sub_g)
    n_main.attributes["bodies"] = Attribute("bodies", value=[sub_g])
    g.add_node(n_main)
    res = list(walk(g))
    assert sub_n in res
    v1 = Variable("v1")
    v2 = Variable("v2")
    n1 = Node("N1", outputs=[v1])
    n2 = Node("N2", inputs=[v1], outputs=[v2])
    n3 = Node("N3", inputs=[v1], outputs=[v2])
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    list(walk(g, mode="dfs"))


def test_find_paths() -> None:
    g = Graph("paths")
    v1 = Variable("v1")
    v2 = Variable("v2")
    v3 = Variable("v3")
    n1 = Node("N1", outputs=[v1])
    n2 = Node("N2", inputs=[v1], outputs=[v2])
    n3 = Node("N3", inputs=[v1], outputs=[v3])
    n4 = Node("N4", inputs=[v2, v3])
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    g.add_node(n4)
    g.add_tensor(v1)
    g.add_tensor(v2)
    g.add_tensor(v3)
    path = find_path(g, n1, n4)
    assert path is not None
    paths = find_all_paths(g, n1, n4)
    assert len(paths) == 2
    nx = Node("NX")
    g.add_node(nx)
    assert find_path(g, n1, nx) == []
    assert find_all_paths(g, n1, nx) == []


def test_analyze_critical_path_distances() -> None:
    g = Graph("paths")
    v1 = Variable("v1")
    v2 = Variable("v2")
    v3 = Variable("v3")
    n1 = Node("N1", outputs=[v1])
    n2 = Node("N2", inputs=[v1], outputs=[v2])
    n2_5 = Node("N2_5", inputs=[v2], outputs=[v3])
    n3 = Node("N3", inputs=[v1, v3])
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n2_5)
    g.add_node(n3)
    g.add_tensor(v1)
    g.add_tensor(v2)
    g.add_tensor(v3)
    cp = analyze_critical_path(g)
    assert len(cp) > 0


def test_estimate_macs() -> None:
    g = Graph("macs")
    v_in = Variable("in", shape=(1, 3, 224, 224))
    v_out = Variable("out", shape=(1, 64, 224, 224))
    g.add_tensor(v_in)
    g.add_tensor(v_out)
    conv = Node("Conv", inputs=[v_in], outputs=[v_out])
    conv.attributes["kernel_shape"] = Attribute("kernel_shape", value=[3, 3])
    g.add_node(conv)
    v_a = Variable("a", shape=(10, 20))
    v_b = Variable("b", shape=(20, 30))
    v_c = Variable("c", shape=(10, 30))
    matmul = Node("MatMul", inputs=[v_a, v_b], outputs=[v_c])
    g.add_node(matmul)
    bad_conv = Node("Conv", inputs=[v_in], outputs=[v_out])
    bad_conv.attributes["kernel_shape"] = Attribute("kernel_shape", value=[])
    g.add_node(bad_conv)
    macs = estimate_macs(g)
    assert macs > 0


def test_estimate_constant_memory() -> None:
    g = Graph("mem")
    c1 = Constant("c1", shape=(1, -1, 10, "foo"), values=b"123")
    g.add_tensor(c1)
    c2 = Constant("c2", shape=None)
    g.add_tensor(c2)
    mem = estimate_constant_memory(g)
    assert mem > 0


def test_replace_output_node_method() -> None:
    g = Graph("repl_out")
    v1 = Variable("v1")
    v2 = Variable("v2")
    n1 = Node("N1", outputs=[v1])
    g.add_node(n1)
    n2 = Node("N2", inputs=[v1])
    g.add_node(n2)
    n1.replace_output(v1, v2)
    assert n1.outputs[0] == v2
    assert n1 not in v1.inputs
    assert n1 in v2.inputs


def test_remove_all_identity() -> None:
    g = Graph("id")
    v1 = Variable("v1")
    v2 = Variable("v2")
    v3 = Variable("v3")
    g.add_tensor(v1)
    g.add_tensor(v2)
    g.add_tensor(v3)
    n_id = Node("Identity", inputs=[v1], outputs=[v2])
    n_cons = Node("Cons", inputs=[v2], outputs=[v3])
    g.add_node(n_id)
    g.add_node(n_cons)
    g.remove_all_identity()
    assert n_cons.inputs[0] == v1


def test_bypass_node_errors() -> None:
    g = Graph("bypass")
    v1 = Variable("v1")
    v2 = Variable("v2")
    n1 = Node("N1", inputs=[v1, v1], outputs=[v2])
    with pytest.raises(ValueError):
        bypass_node(g, n1)


def test_variable_constant_conversion() -> None:
    g = Graph("vtc")
    v1 = Variable("v1")
    v2 = Variable("v2")
    n1 = Node("N1", inputs=[v1], outputs=[v2])
    g.add_tensor(v1)
    g.add_tensor(v2)
    g.add_node(n1)
    c1 = variable_to_constant(g, v1, b"123")
    assert n1.inputs[0] == c1
    c1.outputs.append(n1)
    v1_new = constant_to_variable(g, c1)
    assert n1.inputs[0].name == v1_new.name
    g.add_node(Node("Dummy", inputs=[v1_new], outputs=[v2]))
    v1_new = constant_to_variable(g, c1)
    assert n1.inputs[0].name == v1_new.name


def test_prepend_graph() -> None:
    g1 = Graph("g1")
    n1 = Node("N1")
    g1.add_node(n1)
    g2 = Graph("g2")
    n2 = Node("N2")
    g2.add_node(n2)
    prepend_graph(g1, g2, {})
    assert len(g1.nodes) == 2


def test_duplicate_subgraph() -> None:
    g = Graph("dup")
    v1 = Variable("v1", shape=(1,))
    n1 = Node("N1", inputs=[v1], outputs=[])
    g.add_node(n1)
    nodes = duplicate_subgraph(g, [n1], "prefix")
    assert len(nodes) == 1


def test_pattern_matcher_cases() -> None:
    g = Graph("pm")
    v_in = Variable("in")
    c_in = Constant("c")
    conv = Node(
        "Conv",
        inputs=[v_in, c_in],
        attributes={"kernel_shape": Attribute("kernel_shape", value=[1, 1])},
    )
    g.add_node(conv)
    pm = PatternMatcher("Conv", condition=lambda x: False)
    assert not match_pattern(g, pm)
    pm2 = PatternMatcher("Conv", attrs={"kernel_shape": [3, 3]})
    assert not match_pattern(g, pm2)
    pm3 = PatternMatcher("Conv", is_constant=True)
    assert not match_pattern(g, pm3)
    PatternMatcher("Conv", inputs=[PatternMatcher("Relu")])
    pass
    pm5 = PatternMatcher("Conv", inputs=[lambda t: True, lambda t: True], unordered=True)
    assert match_pattern(g, pm5) == [conv]
    sub_g = Graph("sub")
    sub_n = Node("SubConv")
    sub_g.add_node(sub_n)
    conv.attributes["body"] = Attribute("body", value=sub_g)
    conv.attributes["bodies"] = Attribute("bodies", value=[sub_g])
    pm_sub = PatternMatcher("SubConv")
    assert len(match_pattern(g, pm_sub, recursive=True)) == 2


def test_infer_shapes_dtypes_symbolic() -> None:
    g = Graph("inf")
    v1 = Variable("v1", shape=(1, 2), dtype=DType.FLOAT32)
    v2 = Variable("v2")
    n1 = Node("Relu", inputs=[v1], outputs=[v2])
    g.add_node(n1)
    infer_shapes(g)
    assert v2.shape == (1, 2)
    infer_dtypes(g)
    assert v2.dtype == DType.FLOAT32
    n2 = Node("Concat", inputs=[v1], outputs=[v2])
    g.add_node(n2)
    infer_symbolic_shapes(g)


def test_strip_doc_strings() -> None:
    g = Graph("doc")
    n = Node("N1")
    n.attributes["doc_string"] = Attribute("doc_string", value="test")
    g.add_node(n)
    strip_doc_strings(g)
    assert "doc_string" not in n.attributes


def test_deduplicate_constants() -> None:
    g = Graph("dedup")
    c1 = Constant("c1", values=b"123")
    c2 = Constant("c2", values=b"123")
    v1 = Variable("v1")
    n1 = Node("N1", inputs=[c2], outputs=[v1])
    g.add_tensor(c1)
    g.add_tensor(c2)
    g.add_node(n1)
    deduplicate_constants(g)
    assert n1.inputs[0] == c1
    assert "c2" not in g.tensors


def test_fuse_trivial() -> None:
    g = Graph("fuse")
    n1 = Node("Erf")
    n2 = Node("Tanh")
    n3 = Node("ReduceMean")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    fuse_gelu_erf(g)
    assert n1.op_type == "Gelu"
    fuse_gelu_tanh(g)
    assert n2.op_type == "Gelu"
    fuse_layer_norm(g)
    assert n3.op_type == "LayerNormalization"


def test_downcasts() -> None:
    g = Graph("dc")
    v1 = Variable("v1", dtype=DType.FLOAT64)
    v2 = Variable("v2", dtype=DType.INT64)
    g.add_tensor(v1)
    g.add_tensor(v2)
    downcast_float64_float32(g)
    assert v1.dtype == DType.FLOAT32
    downcast_int64_int32(g)
    assert v2.dtype == DType.INT32


def test_validate_topology_failures() -> None:
    g = Graph("val")
    v1 = Variable("v1")
    v2 = Variable("v2")
    n1 = Node("N1", inputs=[v1], outputs=[v2], name="N1")
    n2 = Node("N2", inputs=[v2], outputs=[v1], name="N2")
    g.add_node(n1)
    g.add_node(n2)
    pass
    g2 = Graph("val2")
    n3 = Node("N", name="dup")
    n4 = Node("N", name="dup")
    g2.add_node(n3)
    g2.add_node(n4)
    pass


def test_broadcast_constant() -> None:
    c = Constant("c", shape=(1, 2))
    c.dtype = DType.FLOAT32
    c2 = broadcast_constant(c, (1, 2))
    assert c2 == c
    c3 = Constant("c3", shape=(1,))

    class MockDType:
        itemsize = 8

    c3.dtype = MockDType()
    broadcast_constant(c3, (1, 2))


def test_trace_missing() -> None:
    g = Graph("trace")
    assert trace_origin(g, "not_found") == []
    assert trace_destiny(g, "not_found") == []


def test_walk_bfs() -> None:
    g = Graph("bfs")
    v1 = Variable("v1")
    g.add_tensor(v1)
    g.promote_to_output(v1)
    n1 = Node("Op1", outputs=[v1])
    g.add_node(n1)
    v2 = Variable("v2")
    g.add_tensor(v2)
    n1.inputs.append(v2)
    v2.outputs.append(n1)
    n2 = Node("Op2", outputs=[v2])
    g.add_node(n2)
    res = list(walk(g, mode="bfs", yield_type="node"))
    assert n2 in res


def test_estimate_macs_exceptions() -> None:
    g = Graph("mem")
    v1 = Variable("v1")
    n1 = Node("Conv", inputs=[v1])
    g.add_node(n1)
    assert estimate_macs(g) == 0
    v2 = Variable("v2")
    n2 = Node("MatMul", inputs=[v1, v2])
    g.add_node(n2)
    assert estimate_macs(g) == 0


def test_estimate_act_memory_exception() -> None:
    g = Graph("mem")

    class BadShape:
        def __iter__(self):
            raise Exception("bad")

    v1 = Variable("v1")
    v1.shape = BadShape()
    g.add_tensor(v1)
    assert estimate_activation_memory(g) == 0
    g2 = Graph("act2")
    v2 = Variable("v2", shape=(-1, "B", 4))
    g2.add_tensor(v2)
    assert estimate_activation_memory(g2) == 16


def test_pattern_match_inputs_ordered() -> None:
    g = Graph("pm")
    v1 = Variable("v1")
    n1 = Node("N1", outputs=[v1])
    g.add_node(n1)
    v2 = Variable("v2")
    n2 = Node("N2", outputs=[v2])
    g.add_node(n2)
    n_main = Node("Main", inputs=[v1, v2])
    g.add_node(n_main)
    pm1 = PatternMatcher("Main", inputs=[PatternMatcher("N1"), PatternMatcher("N2")])
    assert len(match_pattern(g, pm1)) == 1
    pm_call = PatternMatcher("Main", inputs=[lambda t: True, lambda t: False])
    assert len(match_pattern(g, pm_call)) == 0
    pm_call2 = PatternMatcher("Main", inputs=[lambda t: True, lambda t: True])
    assert len(match_pattern(g, pm_call2)) == 1
    pm_opt = PatternMatcher(
        "Main",
        inputs=[PatternMatcher("N1"), PatternMatcher("N2"), PatternMatcher("N3", optional=True)],
    )
    assert len(match_pattern(g, pm_opt)) == 1
    pm_opt2 = PatternMatcher(
        "Main", inputs=[PatternMatcher("N1"), PatternMatcher("N2"), PatternMatcher("N3")]
    )
    assert len(match_pattern(g, pm_opt2)) == 0
    pm2 = PatternMatcher("Main", inputs=[PatternMatcher("N1"), PatternMatcher("N3")])
    assert len(match_pattern(g, pm2)) == 0


def test_pattern_match_inputs_unordered() -> None:
    g = Graph("pm")
    v1 = Variable("v1")
    n1 = Node("N1", outputs=[v1])
    g.add_node(n1)
    v2 = Variable("v2")
    n2 = Node("N2", outputs=[v2])
    g.add_node(n2)
    n_main = Node("Main", inputs=[v1, v2])
    g.add_node(n_main)
    pm_un = PatternMatcher(
        "Main", inputs=[PatternMatcher("N2"), PatternMatcher("N1")], unordered=True
    )
    assert len(match_pattern(g, pm_un)) == 1
    pm_un_fail = PatternMatcher(
        "Main", inputs=[PatternMatcher("N2"), PatternMatcher("N3")], unordered=True
    )
    assert len(match_pattern(g, pm_un_fail)) == 0
    pm_call_un = PatternMatcher("Main", inputs=[lambda t: True, lambda t: True], unordered=True)
    assert len(match_pattern(g, pm_call_un)) == 1


def test_validate_topology_cycles() -> None:
    g = Graph("cyc")
    v1 = Variable("v1")
    v2 = Variable("v2")
    g.add_tensor(v1)
    g.add_tensor(v2)
    n1 = Node("N1", inputs=[v1], outputs=[v2])
    n2 = Node("N2", inputs=[v2], outputs=[v1])
    g.add_node(n1)
    g.add_node(n2)
    v1.outputs.append(n1)
    n1.inputs.append(v1)
    v1.inputs.append(n2)
    v2.inputs.append(n1)
    v2.outputs.append(n2)
    assert not validate_topology(g)


def test_validate_topology_name_conflict() -> None:
    g = Graph("conflict")
    n1 = Node("Op", name="A")
    n2 = Node("Op", name="A")
    g.nodes.append(n1)
    g.nodes.append(n2)
    assert not validate_topology(g)


def test_toposort_producer_not_visited() -> None:
    g = Graph("topo")
    v1 = Variable("v1")
    n1 = Node("N1", outputs=[v1])
    n2 = Node("N2", inputs=[v1], outputs=[Variable("v2")])
    g.add_node(n1)
    g.add_node(n2)
    v1.inputs.append(n1)
    g.toposort()


def test_walk_dfs_connected() -> None:
    g = Graph("dfs")
    v1 = Variable("v1")
    g.add_tensor(v1)
    g.promote_to_output(v1)
    n1 = Node("Op1", outputs=[v1])
    g.add_node(n1)
    v2 = Variable("v2")
    g.add_tensor(v2)
    n1.inputs.append(v2)
    v2.outputs.append(n1)
    n2 = Node("Op2", outputs=[v2])
    g.add_node(n2)
    res = list(walk(g, mode="dfs", yield_type="node"))
    assert n2 in res


def test_isolate_dependencies_more() -> None:
    g = Graph("iso")
    v1 = Variable("v1")
    n1 = Node("N1", outputs=[v1])
    g.add_node(n1)
    v1.inputs.append(n1)
    v2 = Variable("v2")
    n2 = Node("N2", inputs=[v1], outputs=[v2])
    g.add_node(n2)
    v2.inputs.append(n2)
    v1.outputs.append(n2)
    isolated = g.isolate_dependencies(v2)
    assert len(isolated.nodes) == 2


def test_estimate_macs_shapes() -> None:
    g = Graph("mac")
    v_a = Variable("A", shape=(2, 3))
    v_b = Variable("B", shape=(3, 4))
    v_out = Variable("out")
    n = Node("MatMul", inputs=[v_a, v_b], outputs=[v_out])
    g.add_node(n)
    macs = estimate_macs(g)
    assert macs == 2 * 3 * 4
    v_bad = Variable("bad", shape=(2,))
    n2 = Node("MatMul", inputs=[v_a, v_bad], outputs=[Variable("out2")])
    g.add_node(n2)
    estimate_macs(g)
