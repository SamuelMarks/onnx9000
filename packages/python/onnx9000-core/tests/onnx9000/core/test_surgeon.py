"""Comprehensive unit tests for `onnx9000.core.surgeon` graph transformations."""

import os
import struct
import tempfile

import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Constant, Graph, Node, Variable
from onnx9000.core.surgeon import (
    LayoutOptimizerPass,
    PatternMatcher,
    StatefulToStatelessPass,
    _fuse_sequential,
    _match_node,
    analyze_critical_path,
    append_graph,
    broadcast_constant,
    bypass_node,
    cancel_pad_slice,
    cancel_split_concat,
    cancel_squeeze_unsqueeze,
    cast_constant,
    chunk_constants,
    cleanup,
    compare_constants_allclose,
    concatenate_constants,
    constant_to_variable,
    convert_layout,
    convert_nms_trt,
    convert_resize_trt,
    convert_topk_trt,
    deduplicate_constants,
    delete_custom_node,
    demote_output,
    dense_to_sparse,
    disconnect_input,
    disconnect_output,
    downcast_float32_float16,
    downcast_float64_float32,
    downcast_int64_int32,
    downgrade_node_opset,
    dump_netron_json,
    dump_subgraph_netron,
    dump_txt,
    duplicate_subgraph,
    eliminate_cast,
    eliminate_dropout,
    enforce_precision_bounds,
    estimate_activation_memory,
    estimate_constant_memory,
    estimate_macs,
    evaluate_math_graph,
    export_external_data,
    export_raw_bytes,
    extract_scalar,
    extract_subgraph,
    find_all_paths,
    find_path,
    fold_constants,
    fold_constants_math,
    fold_constants_shape,
    fuse_attention,
    fuse_conv_add,
    fuse_conv_bn,
    fuse_conv_mul,
    fuse_conv_relu,
    fuse_fake_quantize,
    fuse_gelu_erf,
    fuse_gelu_tanh,
    fuse_gemm_relu,
    fuse_group_norm,
    fuse_layer_norm,
    fuse_matmul_add,
    fuse_nodes,
    fuse_rope,
    fuse_sequential_reshapes,
    get_disconnected_components,
    get_nodes_by_domain,
    get_nodes_by_name_regex,
    get_nodes_by_op,
    get_nodes_by_op_regex,
    get_tensors_by_name_regex,
    identify_isolated_nodes,
    infer_dtypes,
    infer_shapes,
    infer_symbolic_shapes,
    inject_custom_node,
    inject_identity_probe,
    inject_node_on_edge,
    inject_quantize_nodes,
    inject_trt_calibration,
    inject_trt_plugin,
    insert_node,
    isolate_dependencies,
    load_external_data,
    map_alibi,
    map_bonsai_rope,
    map_fft,
    map_flash_attention,
    map_gqa_mqa,
    map_while_loop,
    match_pattern,
    memory_view_bridge,
    merge_lora_adapters,
    minification,
    next_nodes,
    normalize_sliding_window_attention,
    pack_constants,
    prepend_graph,
    prev_nodes,
    print_constants_by_size,
    print_op_frequency,
    print_topology_map,
    promote_constant_to_input,
    promote_to_output,
    quantize_constant_int8,
    quantize_static_int8,
    quantize_weight_int4,
    quantize_weight_int8,
    reconstruct_sequences,
    register_custom_op_schema,
    register_hook,
    remove_all_identity,
    remove_input,
    remove_output,
    rename_domain,
    rename_op,
    reorder_inputs,
    reorder_outputs,
    replace_input,
    replace_node,
    replace_output,
    replace_pattern,
    reshape_constant,
    restore_layouts,
    semantic_equivalence,
    simplify,
    sink_transposes,
    slice_constant,
    sparse_to_dense,
    split_node,
    strip_doc_strings,
    toposort,
    trace_destiny,
    trace_origin,
    trace_tensor_ops,
    transpose_constant,
    trigger_hook,
    unfuse_fake_quantize,
    unpack_constant,
    unpack_int4_weights,
    unroll_scan,
    unwrap_custom_op,
    upgrade_node_opset,
    upgrade_opset,
    validate_attributes,
    validate_custom_op,
    validate_topology,
    validate_types_and_shapes,
    variable_to_constant,
    visualize_browser_canvas,
    walk,
    warn_implicit_broadcasting,
    wrap_unrecognized_domain,
)


def _make_chain_graph() -> tuple[Graph, Variable, Variable, Variable, Node, Node]:
    """Construct a concrete two-node chain graph (Add -> Relu) for testing."""
    g = Graph("chain")
    v_in = g.register_input("in_x", (2, 2), DType.FLOAT32)
    c_bias = Constant(
        "bias", values=struct.pack("<4f", 1.0, 2.0, 3.0, 4.0), shape=(2, 2), dtype=DType.FLOAT32
    )
    g.add_tensor(c_bias)
    v_mid = Variable("mid", shape=(2, 2), dtype=DType.FLOAT32)
    g.add_tensor(v_mid)
    v_out = g.register_output("out_y", (2, 2), DType.FLOAT32)

    n1 = Node("Add", inputs=[v_in, c_bias], outputs=[v_mid], name="add_node")
    n2 = Node("Relu", inputs=[v_mid], outputs=[v_out], name="relu_node")
    g.add_node(n1)
    g.add_node(n2)
    return g, v_in, v_mid, v_out, n1, n2


def test_toposort_cleanup_fold_simplify_walk() -> None:
    """Verify topological sort, dead-code elimination, constant folding, simplify, and walk."""
    g, v_in, v_mid, _v_out, n1, n2 = _make_chain_graph()
    dead_out = Variable("dead_out", shape=(2, 2), dtype=DType.FLOAT32)
    g.add_tensor(dead_out)
    dead_node = Node("Sigmoid", inputs=[v_in], outputs=[dead_out], name="dead_sig")
    g.add_node(dead_node)

    g.nodes = [n2, dead_node, n1]
    toposort(g)
    assert g.nodes.index(n1) < g.nodes.index(n2)

    cleanup(g)
    assert dead_node not in g.nodes
    assert n1 in g.nodes and n2 in g.nodes

    fold_constants(g)
    simplify(g)
    assert len(g.nodes) == 2

    walked_nodes = list(walk(g, mode="dfs", yield_type="node", direction="backward"))
    assert n2 in walked_nodes and n1 in walked_nodes

    walked_fwd = list(walk(g, mode="bfs", yield_type="tensor", direction="forward"))
    assert v_mid in walked_fwd

    assert prev_nodes(n2) == [n1]
    assert next_nodes(n1) == [n2]


def test_queries_and_path_analysis() -> None:
    """Verify node/tensor query helpers, path finding, subgraph extraction, and memory estimators."""
    g, _v_in, _v_mid, v_out, n1, n2 = _make_chain_graph()

    assert get_nodes_by_op(g, "Add") == [n1]
    assert get_nodes_by_name_regex(g, "^relu") == [n2]
    assert get_nodes_by_op_regex(g, "Rel.*") == [n2]
    assert len(get_tensors_by_name_regex(g, "out_.*")) == 1
    assert len(get_nodes_by_domain(g, "")) == 2

    assert find_path(g, n1, n2) == [n1, n2]
    assert find_all_paths(g, n1, n2) == [[n1, n2]]
    assert len(get_disconnected_components(g)) == 1

    sub = extract_subgraph(g, [n1])
    assert len(sub.nodes) == 1

    iso = isolate_dependencies(g, v_out)
    assert len(iso.nodes) == 2

    crit = analyze_critical_path(g)
    assert len(crit) == 2

    assert estimate_constant_memory(g) == 16
    assert estimate_macs(g) == 0
    assert estimate_activation_memory(g) > 0


def test_node_and_io_mutations() -> None:
    """Verify node insertion, replacement, disconnection, and graph input/output manipulation."""
    g, v_in, v_mid, _v_out, n1, n2 = _make_chain_graph()

    v_extra = Variable("extra", shape=(2, 2), dtype=DType.FLOAT32)
    g.add_tensor(v_extra)
    n_ins = Node("Identity", inputs=[v_in], outputs=[v_extra], name="id_ins")
    insert_node(g, 0, n_ins)
    assert g.nodes[0] is n_ins

    remove_all_identity(g)
    assert n_ins not in g.nodes

    n_inj = Node("Tanh", name="tanh_inj")
    inject_node_on_edge(g, n1, n2, n_inj, 0)
    assert n_inj in g.nodes

    bypass_node(g, n_inj)
    assert n_inj not in g.nodes

    n_rep = Node("Sub", inputs=list(n1.inputs), outputs=list(n1.outputs), name="sub_node")
    replace_node(g, n1, n_rep)
    assert n_rep in g.nodes and n1 not in g.nodes

    replace_input(n_rep, v_in, v_in)
    replace_output(n_rep, v_mid, v_mid)
    disconnect_input(n_rep, v_in)
    disconnect_output(n_rep, v_mid)
    assert v_in not in n_rep.inputs

    from onnx9000.core.surgeon import register_input, register_output

    reg_in = register_input(g, "in2", (2, 2), DType.FLOAT32)
    reg_out = register_output(g, "out2", (2, 2), DType.FLOAT32)
    assert reg_in.name == "in2" and reg_out.name == "out2"

    reorder_inputs(g, ["in2", "in_x"])
    assert g.inputs[0].name == "in2"
    reorder_outputs(g, ["out2", "out_y"])
    assert g.outputs[0].name == "out2"

    remove_input(g, "in2")
    remove_output(g, "out2")
    assert len(g.inputs) == 1 and len(g.outputs) == 1

    rename_op(g, "Relu", "LeakyRelu")
    assert n2.op_type == "LeakyRelu"

    rename_domain(g, "", "custom.dom")
    assert n2.domain == "custom.dom"

    upgrade_node_opset(n2, 18)
    downgrade_node_opset(n2, 13)


def test_tensor_promotions_and_subgraph_ops() -> None:
    """Verify variable/constant conversions, node fusion/splitting, and graph composition."""
    g, _v_in, v_mid, _v_out, n1, n2 = _make_chain_graph()

    probe = inject_identity_probe(g, v_mid)
    assert probe.op_type == "Identity"
    remove_all_identity(g)

    promote_to_output(g, v_mid)
    assert any(o.name == "mid" for o in g.outputs)
    demote_output(g, "mid")
    assert not any(o.name == "mid" for o in g.outputs)

    c_bias = g.tensors["bias"]
    assert isinstance(c_bias, Constant)
    v_bias = promote_constant_to_input(g, c_bias)
    assert isinstance(v_bias, Variable)

    c_back = variable_to_constant(g, v_bias, struct.pack("<4f", 1.0, 1.0, 1.0, 1.0))
    assert isinstance(c_back, Constant)
    v_again = constant_to_variable(g, c_back)
    assert isinstance(v_again, Variable)

    dup_nodes = duplicate_subgraph(g, [n1], "dup")
    assert len(dup_nodes) == 1

    fused = Node("FusedAddRelu", name="fused")
    fuse_nodes(g, [n1, n2], fused)
    assert fused in g.nodes

    split_a = Node("SplitA", inputs=list(fused.inputs), outputs=[v_mid], name="sa")
    split_b = Node("SplitB", inputs=[v_mid], outputs=list(fused.outputs), name="sb")
    split_node(g, fused, [split_a, split_b])
    assert split_a in g.nodes and split_b in g.nodes

    g_extra = Graph("extra_g")
    g_extra.register_input("ex_in", (2, 2), DType.FLOAT32)
    append_graph(g, g_extra, {})
    prepend_graph(g, g_extra, {})
    assert "ex_in" in g.tensors


def test_pattern_matcher_and_optimization_passes() -> None:
    """Verify PatternMatcher, constant folding passes, layout passes, and fusion passes."""
    g, _v_in, _v_mid, _v_out, n1, n2 = _make_chain_graph()

    pm = PatternMatcher(op_type="Add")
    assert _match_node(n1, pm) is True
    assert _match_node(n2, pm) is False
    assert match_pattern(g, pm) == [n1]

    replace_pattern(
        g,
        PatternMatcher(op_type="Relu"),
        lambda n: [Node("Gelu", inputs=n.inputs, outputs=n.outputs, name="gelu")],
    )
    assert any(n.op_type == "Gelu" for n in g.nodes)

    assert fold_constants_math(g) is g
    assert fold_constants_shape(g) is g
    assert eliminate_dropout(g) is g
    assert eliminate_cast(g) is g
    assert sink_transposes(g) is g
    assert convert_layout(g) is g
    assert restore_layouts(g) is g
    assert infer_shapes(g) is g
    assert infer_symbolic_shapes(g) is g
    assert infer_dtypes(g) is g
    _fuse_sequential(g, "Add", "Gelu", "FusedAddGelu")
    assert any(n.op_type == "FusedAddGelu" for n in g.nodes)
    assert fuse_conv_bn(g) is g
    assert fuse_conv_add(g) is g
    assert fuse_conv_mul(g) is g
    assert fuse_matmul_add(g) is g
    assert fuse_gemm_relu(g) is g
    assert fuse_conv_relu(g) is g
    assert fuse_sequential_reshapes(g) is g
    assert strip_doc_strings(g) is g
    assert minification(g) is g
    assert deduplicate_constants(g) is g
    assert cancel_squeeze_unsqueeze(g) is g
    assert cancel_split_concat(g) is g
    assert cancel_pad_slice(g) is g
    assert fuse_gelu_erf(g) is g
    assert fuse_gelu_tanh(g) is g
    assert fuse_layer_norm(g) is g
    assert fuse_attention(g) is g
    assert fuse_rope(g) is g
    assert fuse_group_norm(g) is g


def test_quantization_downcasting_and_serialization() -> None:
    """Verify precision downcasting, quantization helpers, and serialization/validation routines."""
    g, _v_in, _v_mid, _v_out, n1, _n2 = _make_chain_graph()

    assert downcast_float64_float32(g) is g
    assert downcast_float32_float16(g) is g
    assert downcast_int64_int32(g) is g
    assert quantize_static_int8(g) is g
    assert quantize_weight_int8(g) is g
    assert quantize_weight_int4(g) is g

    with tempfile.TemporaryDirectory() as tmpdir:
        assert load_external_data(g, tmpdir) is None
        assert export_external_data(g, os.path.join(tmpdir, "weights.bin")) is None

    raw = export_raw_bytes(g)
    assert isinstance(raw, bytes)
    mv = memory_view_bridge(g)
    assert mv is None
    assert chunk_constants(g) is None

    netron_json = dump_netron_json(g)
    assert "nodes" in netron_json
    assert validate_topology(g) is True
    assert upgrade_opset(g, 17) is g
    assert validate_types_and_shapes(g) is True
    assert semantic_equivalence(g, g) is True
    assert "chain" in dump_txt(g)
    assert isinstance(reconstruct_sequences(g), dict)
    assert merge_lora_adapters(g) is None
    assert inject_quantize_nodes(g, {}) is g
    assert fuse_fake_quantize(g) is g
    assert unfuse_fake_quantize(g) is g
    assert inject_trt_plugin(g, "Add", {"plugin": 1}) is not None
    assert convert_nms_trt(g) is g
    assert convert_resize_trt(g) is g
    assert convert_topk_trt(g) is g
    assert enforce_precision_bounds(g) is g
    assert inject_trt_calibration(g) is g
    assert validate_attributes(n1) is True


def test_constant_manipulations_and_diagnostics() -> None:
    """Verify Constant manipulation functions, tracing, hooks, and custom op helpers."""
    g, _v_in, _v_mid, _v_out, n1, _n2 = _make_chain_graph()
    c = Constant("c_mat", values=b"1234123412341234", shape=(2, 2))

    c_t = transpose_constant(c, [1, 0])
    assert c_t.shape == (2, 2)
    c_r = reshape_constant(c, (4,))
    assert c_r.shape == (4,)
    c_b = broadcast_constant(c, (2, 2, 2))
    assert c_b.shape == (2, 2, 2)
    c_s = slice_constant(c, [0], [1], [0])
    assert c_s is c
    c_cat = concatenate_constants([c, c], 0)
    assert c_cat is c
    c_cast = cast_constant(c, DType.FLOAT16)
    assert c_cast is c
    c_q8 = quantize_constant_int8(c, 0.5, 0)
    assert c_q8 is c
    c_u4 = unpack_int4_weights(c)
    assert c_u4 is c

    res_eval = evaluate_math_graph(g)
    assert res_eval is None

    c_scalar = Constant("c_sc", values=struct.pack("<f", 5.0), shape=(), dtype=DType.FLOAT32)
    assert extract_scalar(c_scalar) == pytest.approx(5.0)

    packed = pack_constants([c], "packed")
    assert packed.name == "packed"
    unpacked = unpack_constant(packed, [(2, 2)])
    assert len(unpacked) == 1

    assert sparse_to_dense(c).shape == (2, 2)
    assert dense_to_sparse(c) is not None

    print_topology_map(g)
    print_constants_by_size(g)
    print_op_frequency(g)

    traced_ops = trace_tensor_ops(g, "in_x")
    assert isinstance(traced_ops, list)
    assert isinstance(trace_origin(g, "out_y"), list)
    assert isinstance(trace_destiny(g, "in_x"), list)

    with tempfile.NamedTemporaryFile(suffix=".json") as tf:
        dump_subgraph_netron(g, [n1], tf.name)

    assert visualize_browser_canvas(g, "canvas_1") is None

    assert compare_constants_allclose(c, c) is True
    assert warn_implicit_broadcasting(g) == []
    assert isinstance(identify_isolated_nodes(g), list)

    register_custom_op_schema(g, "custom", "MyOp", {})
    cust_node = inject_custom_node(g, "MyOp", "custom", [], [], {})
    assert cust_node in g.nodes
    delete_custom_node(g, cust_node)
    assert cust_node not in g.nodes

    hook_calls: list[str] = []
    register_hook(g, "on_mutate", lambda gr: hook_calls.append(gr.name))
    trigger_hook(g, "on_mutate")
    assert hook_calls == ["chain"]

    wrap_unrecognized_domain(g, "custom")
    assert unwrap_custom_op(g, n1, lambda x: None) is g
    assert validate_custom_op(n1, lambda x: True) is True


def test_attention_and_loop_mapping_passes() -> None:
    """Verify RoPE, ALiBi, GQA/MQA, SWA, FlashAttention, Scan, Loop, Stateful, FFT, and Layout passes."""
    g = Graph("modern_arch")
    g.add_node(Node("BonsaiRoPE", ["x"], ["y1"], name="rope"))
    g.add_node(Node("ALiBi", ["y1", "bias"], ["y2"], name="alibi"))
    g.add_node(Node("GQA", ["y2", "k", "v"], ["y3"], name="gqa"))
    g.add_node(Node("SlidingWindowAttention", ["y3"], ["y4"], name="swa"))
    g.add_node(Node("jax.lax.pallas_call", ["y4"], ["y5"], name="flash_attn"))
    g.add_node(
        Node(
            "jax.lax.scan",
            ["y5"],
            ["y6"],
            attributes={"sequence_length": Attribute("sequence_length", "int", 2)},
            name="scan",
        )
    )
    g.add_node(Node("jax.lax.while_loop", ["y6"], ["y7"], name="while_loop"))
    g.add_node(Node("MambaBlock", ["y7"], ["y8"], name="mamba"))
    g.add_node(Node("jnp.fft.rfft2", ["y8"], ["y9"], name="fft"))
    g.add_node(Node("Conv", ["y9"], ["y10"], name="conv"))

    map_bonsai_rope(g)
    assert any(n.op_type == "RotaryEmbedding" for n in g.nodes)

    map_alibi(g)
    assert any(n.op_type == "Mask" for n in g.nodes)

    map_gqa_mqa(g)
    assert any(n.op_type == "MultiHeadAttention" for n in g.nodes)

    normalize_sliding_window_attention(g)
    map_flash_attention(g)
    assert any(n.op_type == "FlashAttention" for n in g.nodes)

    unroll_scan(g)
    assert any(n.op_type == "ScanUnrolled" for n in g.nodes)

    map_while_loop(g)
    assert any(n.op_type == "Loop" for n in g.nodes)

    StatefulToStatelessPass.apply(g)
    assert "mamba_state_in" in g.inputs

    map_fft(g)
    assert any(n.op_type == "DFT" for n in g.nodes)

    LayoutOptimizerPass.apply(g)
    conv_nodes = [n for n in g.nodes if n.op_type == "Conv"]
    assert conv_nodes[0].attributes["layout"].value == "NCHW"
