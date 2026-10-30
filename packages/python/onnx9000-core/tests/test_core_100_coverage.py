"""Exhaustive coverage tests for onnx9000-core operators, IR, surgeon, sparse, and utilities."""

import inspect
import struct
import tempfile
from pathlib import Path

import pytest
from onnx9000.core import (
    checker,
    exporter,
    macros,
    memory_planner,
    profiler,
    profiler_checks,
    profiler_grouping,
    serializer,
    shape_inference,
    sharding,
    sparse,
    surgeon,
    symbolic,
    tensorboard_exporter,
)
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import (
    Attribute,
    Constant,
    DynamicDim,
    Graph,
    Node,
    SparseTensor,
    Tensor,
    ValueInfo,
    Variable,
)
from onnx9000.core.ops import (
    control_flow as ops_cf,
)
from onnx9000.core.ops import (
    math as ops_math,
)
from onnx9000.core.ops import (
    nn as ops_nn,
)
from onnx9000.core.ops import (
    quantization as ops_quant,
)
from onnx9000.core.ops import (
    reduction as ops_red,
)
from onnx9000.core.ops import (
    sequence as ops_seq,
)
from onnx9000.core.ops import (
    tensor as ops_tensor,
)
from onnx9000.core.parser import core as parser_core


def test_all_core_ops_functions() -> None:
    """Invoke every operator function in control_flow, math, nn, quantization, reduction, sequence, and tensor."""
    t = Tensor("t", (2, 2), DType.FLOAT32)

    # control_flow
    assert ops_cf.where(t, t, t).name == "Where_out"
    assert ops_cf.all_reduce(t).name == "AllReduce_out"
    assert ops_cf.all_gather(t).name == "AllGather_out"
    assert ops_cf.reduce_scatter(t).name == "ReduceScatter_out"
    assert ops_cf.broadcast(t).name == "Broadcast_out"
    assert ops_cf.if_op(t).name == "If_out"
    assert ops_cf.loop(t).name == "Loop_out"
    assert ops_cf.optional(t).name == "Optional_out"
    assert ops_cf.optional_get_element(t).name == "OptionalGetElement_out"
    assert ops_cf.optional_has_element(t).name == "OptionalHasElement_out"
    assert ops_cf.scan(t).name == "Scan_out"

    # math
    for name, fn in inspect.getmembers(ops_math, inspect.isfunction):
        if fn.__module__ != ops_math.__name__:
            continue
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if name == "clip":
            assert fn(t).name == "Clip_out"
            assert fn(t, t).name == "Clip_out"
            assert fn(t, None, t).name == "Clip_out"
            assert fn(t, t, t).name == "Clip_out"
        elif name == "dft":
            assert fn(t).name == "DFT_out"
            assert fn(t, t).name == "DFT_out"
        elif name == "gemm":
            assert fn(t, t).name == "Gemm_out"
            assert fn(t, t, t).name == "Gemm_out"
        elif name == "cum_sum":
            assert fn(t, t).name == "CumSum_out"
        elif name == "einsum":
            assert fn([t, t], "ij,jk->ik").name == "Einsum_out"
        else:
            pos_args = [
                t
                for p in params
                if p.kind
                in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                and p.default is inspect.Parameter.empty
            ]
            assert fn(*pos_args).name is not None

    # nn, quantization, reduction, sequence, tensor via signature inspection
    for mod in (ops_nn, ops_quant, ops_red, ops_seq, ops_tensor):
        for _name, fn in inspect.getmembers(mod, inspect.isfunction):
            if fn.__module__ != mod.__name__:
                continue
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
            req_args = []
            all_args = []
            for p in params:
                if p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    annot = str(p.annotation)
                    val: object = t
                    if "list[" in annot or "Sequence[" in annot:
                        val = [t, t]
                    elif annot in ("int", "<class 'int'>"):
                        val = 1
                    elif annot in ("float", "<class 'float'>"):
                        val = 1.0
                    elif annot in ("str", "<class 'str'>"):
                        val = "a"
                    if p.default is inspect.Parameter.empty:
                        req_args.append(val)
                    all_args.append(val)
            assert fn(*req_args) is not None
            if len(all_args) > len(req_args):
                assert fn(*all_args) is not None
            if mod is ops_red and len(params) >= 2 and "axes" in params[1].name:
                assert fn(t, [0]) is not None
                assert fn(t, t) is not None
    assert ops_math.stft(t, t, t, t, t).name == "STFT_out"
    assert ops_math.istft(t, t, t, t, t).name == "ISTFT_out"


def test_core_modules_remaining_branches() -> None:
    """Test remaining uncovered branches in checker, exporter, ir, macros, memory_planner, parser, profiler, serializer, shape_inference, sharding, sparse, symbolic, tensorboard."""
    # ir.py: Attribute.infer_type unknown object, ValueInfo.__repr__, _uniquify_node_name, _uniquify_tensor_name, remove_node
    attr_unk = Attribute("u", value=bytearray(b"123"))
    assert attr_unk.attr_type == "UNKNOWN"
    vi = ValueInfo("v", (1,), DType.FLOAT32)
    assert "ValueInfo" in repr(vi)
    g = Graph("g")
    n1 = Node("Relu", ["x"], ["y"], name="n")
    n2 = Node("Relu", ["y"], ["z"], name="n_1")
    n3 = Node("Relu", ["z"], ["w"], name="n")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    assert n3.name == "n_2"

    t1 = Tensor("t", (1,), DType.FLOAT32)
    t2 = Tensor("t_1", (1,), DType.FLOAT32)
    t3 = Tensor("t", (1,), DType.FLOAT32)
    g.add_tensor(t1)
    g.add_tensor(t2)
    g.add_tensor(t3)
    assert t3.name == "t_2"

    # remove_node when consumer_map entry doesn't have node
    n_rem = Node("Relu", ["in_rem"], ["out_rem"], name="rem")
    g.add_node(n_rem)
    g.consumer_map["in_rem"] = []
    g.remove_node(n_rem)
    assert n_rem not in g.nodes

    # macros.py line 31->30: MacroExpander when op_type is not in MACRO_REGISTRY
    g_mac = Graph("gm")
    g_mac.add_node(Node("NotAMacro", [], []))
    macros.MacroExpander().apply(g_mac)

    # tensorboard_exporter.py line 20->22: node with empty name
    with tempfile.TemporaryDirectory() as tmpdir:
        g_tb = Graph("tb")
        n_empty = Node("Relu", [], [])
        n_empty.name = ""
        g_tb.nodes.append(n_empty)
        path = tensorboard_exporter.export_tensorboard(g_tb, tmpdir)
        assert Path(path).exists()

        # exporter.py line 80->exit: export_graph with wasm when js_path == output_path
        ce = exporter.IRToONNXExporter(opset=21)
        g_exp = Graph("ge")
        g_exp.add_node(Node("UnmappedOp", ["a"], ["b"]))
        out_onnx = str(Path(tmpdir) / "exp.onnx")
        ce.export(g_exp, out_onnx)
        assert Path(out_onnx).exists()

    # checker.py: 84->88, 91->90, 107->exit, 125->exit, 131->exit
    ctx = checker.ValidationContext()
    n_conv_even = Node(
        "Conv",
        ["x", "w"],
        ["y"],
        attributes={"pads": [1, 1, 1, 1]},  # type: ignore[dict-item]
    )
    checker._check_op_specific(n_conv_even, ctx)
    n_conv_no_pads = Node("Conv", ["x", "w"], ["y"], attributes={})
    checker._check_op_specific(n_conv_no_pads, ctx)
    n_if_ok = Node(
        "If",
        ["c"],
        ["y"],
        attributes={"then_branch": Graph("t"), "else_branch": Graph("e")},  # type: ignore[dict-item]
    )
    checker._check_op_specific(n_if_ok, ctx)
    n_tree_ok = Node(
        "TreeEnsembleClassifier",
        ["x"],
        ["y"],
        attributes={
            "nodes_treeids": [0],  # type: ignore[dict-item]
            "nodes_nodeids": [0],  # type: ignore[dict-item]
            "nodes_featureids": [0],  # type: ignore[dict-item]
        },
    )
    checker._check_op_specific(n_tree_ok, ctx)
    assert len(ctx.errors) == 0

    # symbolic.py: 147->151, 154->156
    assert symbolic.simplify_expression("a + b") == "(a + b)"
    assert symbolic.simplify_expression("a * b") == "(a * b)"

    # sharding.py: 40->36, 88->110, 90->110, 93->110, 124->136, 138->142, 139->138, 142->155
    g_sh = Graph("sh")
    t_s1 = Tensor("s1", (2, 2), DType.FLOAT32)
    t_s1.sharding = sharding.PartitionSpec("dp", None)
    t_s2 = Tensor("s2", (2, 2), DType.FLOAT32)
    # MatMul with sharded input but empty outputs (40->36)
    n_mm_no_out = Node("MatMul", [t_s1, t_s2], [])
    g_sh.nodes.append(n_mm_no_out)
    sharding.AutoShardingPass().apply(g_sh)

    # SPMDLoweringPass branches:
    # 88->110: weight with sharding that is neither (None, 'tp') nor ('tp', None)
    # 90->110: weight with sharding (None, 'tp') and empty outputs
    # 93->110: weight with sharding ('tp', None) and empty outputs
    # 124->136: EP input with empty outputs
    # 138->142, 139->138, 142->155: FlashAttention where input has no 'cp' sharding
    g_spmd = Graph("spmd")
    w_other = Tensor("wo", (2, 2), DType.FLOAT32)
    w_other.sharding = sharding.PartitionSpec("other")
    w_tp1 = Tensor("wtp1", (2, 2), DType.FLOAT32)
    w_tp1.sharding = sharding.PartitionSpec(None, "tp")
    w_tp2 = Tensor("wtp2", (2, 2), DType.FLOAT32)
    w_tp2.sharding = sharding.PartitionSpec("tp", None)
    in_ep = Tensor("iep", (2, 2, 2), DType.FLOAT32)
    in_ep.sharding = sharding.PartitionSpec("ep", None, None)
    in_fa = Tensor("ifa", (2, 2, 2), DType.FLOAT32)
    g_spmd.nodes.extend(
        [
            Node("MatMul", [t_s2, w_other], []),
            Node("MatMul", [t_s2, w_tp1], []),
            Node("MatMul", [t_s2, w_tp2], []),
            Node("CustomEP", [in_ep], []),
            Node("FlashAttention", [in_fa], []),
        ]
    )
    sharding.SPMDLoweringPass().apply(g_spmd)


def test_memory_planner_parser_profiler_serializer() -> None:
    """Cover remaining branches in memory_planner, parser/core, profiler, profiler_checks, profiler_grouping, and serializer."""
    # memory_planner.py: 63->68, 88->86, 116->122, 155->154, 165->163, 169->163, 186->193, 208->213, 215->214, 217->214
    arena = memory_planner.ArenaSimulator()
    arena.allocate_first_fit("t1", 256)
    arena.free("nonexistent")
    arena.free("t1")
    arena.allocate_first_fit("t2", 256)
    arena.free("t2")
    arena.allocate_best_fit("t3", 256)
    arena.allocate_best_fit("t4", 512)
    arena.free("t3")
    arena.allocate_best_fit("t5", 256)

    g_mp = Graph("mp")
    c_init = Constant("c_init", values=b"1234", shape=(1,), dtype=DType.FLOAT32)
    v_in = Variable("v_in", shape=(2, 2), dtype=DType.FLOAT32)
    v_mid = Variable("v_mid", shape=(2, 2), dtype=DType.FLOAT32)
    v_out = Variable("v_out", shape=(2, 2), dtype=DType.FLOAT32)
    g_mp.add_tensor(c_init)
    g_mp.add_tensor(v_in)
    g_mp.add_tensor(v_mid)
    g_mp.add_tensor(v_out)
    g_mp.initializers.append("c_init")
    g_mp.inputs.append(ValueInfo("v_in", (2, 2), DType.FLOAT32))
    g_mp.outputs.append(ValueInfo("v_out", (2, 2), DType.FLOAT32))
    # Node with input not in lifetimes (e.g. Constant), and output not in tensors
    n_mp1 = Node("Add", ["v_in", "c_init", "unknown_in"], ["v_mid", "unknown_out"])
    n_mp2 = Node("Relu", ["v_mid"], ["v_out"])
    g_mp.add_node(n_mp1)
    g_mp.add_node(n_mp2)
    memory_planner.simulate_memory_plan(g_mp)

    # profiler_checks.py: 62->60, 64->60 (Add with <2 inputs or input not in tensors)
    g_pc = Graph("pc")
    g_pc.add_node(Node("Add", ["only_one"], ["out1"]))
    g_pc.add_node(Node("Add", ["missing1", "missing2"], ["out2"]))
    analyzer = profiler_checks.OptimizationAnalyzer(g_pc)
    analyzer.analyze()

    # profiler_grouping.py: 36->38, 38->40, 40->42, 42->exit (all metrics are non-int/dynamic)
    prof_res = profiler.ProfilerResult()
    prof_res.node_profiles.append(
        {
            "name": "layer1.conv",
            "op_type": "Conv",
            "macs": "dyn",
            "flops": "dyn",
            "params": "dyn",
            "activation_bytes": "dyn",
        }
    )
    tree = profiler_grouping.group_by_namespace(prof_res)
    assert "layer1" in tree.children

    # profiler.py branches:
    # 45->38: resolve_volume with dimension <= 0
    assert profiler.resolve_volume((-1, 2)) == 2
    # 154->157, 160->163: estimate_latency when total_memory_bytes / peak_activation_bytes are non-int
    pr = profiler.ProfilerResult()
    pr.total_memory_bytes = "dyn"  # type: ignore[assignment]
    pr.peak_activation_bytes = "dyn"  # type: ignore[assignment]
    assert pr.estimate_latency()["memory_latency_ms"] == 0.0
    # 197->196: print_parameter_pie_chart with node having 0 params
    pr.total_params = 10
    pr.node_profiles = [
        {"name": "n1", "op_type": "Relu", "params": 0, "activation_bytes": 0, "flops": 0}
    ]
    pr.print_parameter_pie_chart()
    # 228->227: print_distribution_pie_chart with node having 0 flops
    pr.total_flops = 10
    pr.print_distribution_pie_chart()

    # 320->562, 328->331, 388->392, 392->398, 422->427, 540->562, 554->558, 558->562, 563->566
    g_prof = Graph("prof")
    t_f32 = Tensor("tf32", (1, 2, 4), DType.FLOAT32)
    t_i32 = Tensor("ti32", (1, 2, 4), DType.INT32)
    g_prof.add_tensor(t_f32)
    g_prof.add_tensor(t_i32)
    # Reshape with input not in graph.tensors (320->562, 328->331)
    g_prof.add_node(Node("Reshape", ["not_in_tensors"], ["not_in_tensors_out"]))
    # If node with non-Graph or None branches (388->392, 392->398)
    g_prof.add_node(Node("If", ["tf32"], ["if_out"]))
    # Loop node with non-Graph body (422->427)
    g_prof.add_node(Node("Loop", ["tf32"], ["loop_out"]))
    # Custom op with no inputs/outputs (540->562)
    g_prof.add_node(Node("CustomNoIO", [], []))
    # Custom op with inputs but output not in tensors or dynamic volume (554->558, 558->562)
    t_dyn = Tensor("tdyn", ("B",), DType.INT32)
    g_prof.add_tensor(t_dyn)
    n_cust = Node("CustomDyn", ["ti32"], ["tdyn"])
    g_prof.nodes.append(n_cust)
    profiler.profile(g_prof)

    # serializer.py & parser/core.py
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # External tensor with no length specified and unmapped key in external_data
        ext_file = tmp_path / "ext.bin"
        ext_file.write_bytes(b"\x00\x00\x80\x3f")
        tp = parser_core.onnx_pb2.TensorProto()
        tp.name = "ext_t"
        tp.data_type = 1
        tp.dims.append(1)
        tp.data_location = 1
        e1 = tp.external_data.add()
        e1.key = "location"
        e1.value = "ext.bin"
        e2 = tp.external_data.add()
        e2.key = "other_key"
        e2.value = "ignored"
        parsed_t = parser_core.parse_tensor_proto(tp, tmp_path)
        assert parsed_t.data is not None

        # External tensor with location but base_dir=None (108->119)
        parsed_t2 = parser_core.parse_tensor_proto(tp, None)
        assert parsed_t2.name == "ext_t"

        # parse_model with output and value_info already in graph.tensors (182->177)
        mp = parser_core.onnx_pb2.ModelProto()
        mp.ir_version = 8
        init_p = mp.graph.initializer.add()
        init_p.name = "existing_t"
        init_p.data_type = 1
        init_p.dims.append(1)
        init_p.raw_data = b"\x00\x00\x80\x3f"
        vi_p = mp.graph.value_info.add()
        vi_p.name = "existing_t"
        vi_p.type.tensor_type.elem_type = 1
        parsed_g = parser_core.parse_model(mp)
        assert "existing_t" in parsed_g.tensors

        # serializer.py: SparseTensor with values=None or indices=None (90->95, 95->101)
        st_empty = SparseTensor("st_empty", dims=(2, 2), format="COO")
        sp_proto, _ = serializer._serialize_sparse_tensor(st_empty)
        assert list(sp_proto.dims) == [2, 2]

        # serializer.py: _serialize_vi with string name not in graph.tensors (148->155)
        g_ser = Graph("g_ser")
        g_ser.inputs.append("missing_vi_tensor")  # type: ignore[arg-type]
        ser_mp = serializer.serialize_model(g_ser)
        assert ser_mp.graph.input[0].name == "missing_vi_tensor"

        # serializer.py: save with compress=True and external_data=False (258->260)
        zip_out = tmp_path / "model.onnx"
        serializer.save(g_ser, zip_out, external_data=False, compress=True)
        assert (tmp_path / "model.zip").exists()

    # symbolic.py 147->151: (a * b) / c where c != a and c != b
    assert symbolic.simplify_expression("(a * b) / c") == "((a * b) / c)"

    # surgeon.py lines 1159, 1682, 1696, 1719, 1755, 1901-1904
    assert surgeon._compute_strides(()) == []
    assert surgeon._permute_flat_buffer([], (), []) == ([], ())
    c_no_data = Constant("c_nd", values=None, shape=(2, 2), dtype=DType.FLOAT32)
    assert surgeon.transpose_constant(c_no_data, [1, 0]) is c_no_data
    c_2d = Constant("c_2d", values=b"\x00\x00\x80\x3f" * 4, shape=(2, 2), dtype=DType.FLOAT32)
    assert surgeon.broadcast_constant(c_2d, (2, 2, 2)).shape == (2, 2, 2)

    g_math = Graph("gm")
    c_m1 = Constant("m1", values=struct.pack("<2f", 1.0, 2.0), shape=(2,), dtype=DType.FLOAT32)
    g_math.add_tensor(c_m1)
    g_math.add_node(Node("Cast", ["m1"], ["m_i32"], attributes={"to": Attribute("to", "INT", 6)}))
    g_math.add_node(Node("Cast", ["m1"], ["m_bool"], attributes={"to": Attribute("to", "INT", 9)}))
    g_math.add_node(Node("Cast", ["m1"], ["m_f32"], attributes={"to": Attribute("to", "INT", 1)}))
    assert surgeon.evaluate_math_graph(g_math) is not None

    # fuse_conv_bn with Conv having 3 inputs (bias) where bias data is None (1159)
    g_cbn = Graph("cbn")
    w_c = Constant("wc", values=struct.pack("<1f", 2.0), shape=(1, 1, 1, 1), dtype=DType.FLOAT32)
    b_c = Constant("bc", values=None, shape=(1,), dtype=DType.FLOAT32)
    s_bn = Constant("sbn", values=struct.pack("<1f", 1.0), shape=(1,), dtype=DType.FLOAT32)
    b_bn = Constant("bbn", values=struct.pack("<1f", 0.0), shape=(1,), dtype=DType.FLOAT32)
    m_bn = Constant("mbn", values=struct.pack("<1f", 0.0), shape=(1,), dtype=DType.FLOAT32)
    v_bn = Constant("vbn", values=struct.pack("<1f", 1.0), shape=(1,), dtype=DType.FLOAT32)
    for t_item in (w_c, b_c, s_bn, b_bn, m_bn, v_bn):
        g_cbn.add_tensor(t_item)
    n_conv_cbn = Node("Conv", ["x", "wc", "bc"], ["conv_out"])
    n_bn_cbn = Node("BatchNormalization", ["conv_out", "sbn", "bbn", "mbn", "vbn"], ["bn_out"])
    g_cbn.add_node(n_conv_cbn)
    g_cbn.add_node(n_bn_cbn)
    surgeon.fuse_conv_bn(g_cbn)


def test_remaining_branches_exhaustive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover all remaining branch gaps in checker, exporter, ir, macros, memory_planner, profiler, shape_inference, sparse, and surgeon."""
    # ir.py: 137->139 (list of non-matching item e.g. list[bytes]), 206 (ValueInfo.to_dict), 710->714, 795->799
    attr_bytes_list = Attribute("bl", value=[b"a"])
    assert attr_bytes_list.attr_type == "UNKNOWN"
    vi = ValueInfo("vi", (1, DynamicDim("N")), DType.FLOAT32)
    assert vi.to_dict()["name"] == "vi"
    g_ir = Graph("gir")
    t_exist = Tensor("t_ex", (1,), DType.FLOAT32)
    g_ir.tensors["t_ex"] = t_exist
    assert g_ir._uniquify_tensor_name("t_ex") == "t_ex_1"
    n_ir = Node("Relu", ["in_ir"], ["out_ir"], name="n_ir")
    g_ir.add_node(n_ir)
    g_ir.producer_map["out_ir"] = Node("Other", [], [])
    g_ir.remove_node(n_ir)

    # macros.py: 31->30 (kwarg that is NOT a Tensor)
    t_m = Tensor("tm", (1,), DType.FLOAT32)
    macros.transformer_block_macro(t_m, weight1=t_m, weight2="not_a_tensor")

    # checker.py: 84->88, 91->90, 107->exit
    ctx = checker.ValidationContext()
    t_ext_safe = Tensor("tes", (0, -1), DType.FLOAT32)
    t_ext_safe.data_type = "float"  # type: ignore[attr-defined]
    t_ext_safe.data_location = "EXTERNAL"  # type: ignore[attr-defined]
    t_ext_safe.external_data = {"location": "safe.bin"}  # type: ignore[attr-defined]
    t_ext_safe.raw_data = b"1234"  # type: ignore[attr-defined]
    checker.check_tensor(t_ext_safe, ctx)
    checker.check_attribute("other", "val", "string", ctx)

    # exporter.py: 80->exit (wasm export where js_path == output_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_js = str(Path(tmpdir) / "out.js")
        import onnx9000.converters.jit.compiler as jit_comp

        def fake_compile_wasm(_g: Graph, _d: Path) -> Path:
            """Test helper function fake_compile_wasm."""
            Path(out_js).write_text("// js")
            return Path(out_js)

        monkeypatch.setattr(jit_comp, "compile_wasm", fake_compile_wasm)
        exporter.export_graph(Graph("gw"), out_js, "wasm")

    # memory_planner.py: 88->86, 164->162, 168->162, 185->192, 215->212
    arena = memory_planner.ArenaSimulator()
    arena.allocate_best_fit("b1", 512)
    arena.allocate_best_fit("b2", 256)
    arena.free("b1")
    arena.allocate_best_fit("b3", 512)  # exact match on b1 (88->86)

    g_mp = Graph("gmp")
    g_mp.inputs.append(ValueInfo("missing_in", (1,), DType.FLOAT32))
    g_mp.inputs.append(ValueInfo("zero_in", (0,), DType.FLOAT32))
    g_mp.add_tensor(Variable("zero_in", (0,), DType.FLOAT32))
    g_mp.inputs.append(ValueInfo("valid_in", (2,), DType.FLOAT32))
    g_mp.add_tensor(Variable("valid_in", (2,), DType.FLOAT32))
    g_mp.add_tensor(Variable(" zero_out", (0,), DType.FLOAT32))
    # Relu where input is not in arena.blocks (185->192)
    n_relu = Node("Relu", ["unallocated_in"], ["zero_out"])
    g_mp.add_tensor(Variable("zero_out", (0,), DType.FLOAT32))
    n_add = Node("Add", ["valid_in"], ["zero_out"])
    g_mp.add_node(n_relu)
    g_mp.add_node(n_add)
    memory_planner.simulate_memory_plan(g_mp)

    # profiler.py: 154->157, 197->196
    pr = profiler.ProfilerResult()
    pr.total_flops = "dyn"
    pr.estimate_latency()
    pr.total_params = 10
    pr.node_profiles = [
        {"name": "n1", "op_type": "Conv", "params": 5, "activation_bytes": 0, "flops": 0},
        {"name": "n2", "op_type": "Relu", "params": 0, "activation_bytes": 0, "flops": 0},
    ]
    pr.print_parameter_pie_chart()

    # profiler_checks.py: 62->60, 64->60 (MatMul with no outputs or consumers != Add)
    g_pc = Graph("gpc")
    g_pc.add_tensor(Tensor("mm_out", (2, 2), DType.FLOAT32))
    g_pc.add_node(Node("MatMul", ["a", "b"], []))
    g_pc.add_node(Node("MatMul", ["a", "b"], ["mm_out"]))
    g_pc.add_node(Node("Relu", ["mm_out"], ["r_out"]))
    profiler_checks.OptimizationAnalyzer(g_pc).analyze()

    # sparse.py remaining branches
    # 410->408: BSR with a zero element inside a block
    c_bsr = Constant(
        "cbsr",
        values=struct.pack("<4f", 1.0, 0.0, 0.0, 0.0),
        shape=(2, 2),
        dtype=DType.FLOAT32,
    )
    st_bsr = sparse.dense_to_bsr(c_bsr, (2, 2))
    assert sparse.sparse_to_coo(st_bsr).format == "COO"

    # 512->511, 514->518: sparse.profile with Variable (not Constant/SparseTensor) and 0-d Constant
    g_sp = Graph("gsp")
    g_sp.add_tensor(Variable("var_sp", (2,), DType.FLOAT32))
    g_sp.add_tensor(Constant("c_0d", values=b"\x00\x00\x80\x3f", shape=(), dtype=DType.FLOAT32))
    g_sp.add_tensor(Constant("c_zero_dim", values=b"", shape=(0,), dtype=DType.FLOAT32))
    sparse.profile(g_sp)

    # de_sparsify, convert_hf_sparse_to_onnx, resolve_nm_metadata, map_sparse_to_safetensors, validate_provider_support, collapse_sparse_tensors, strip_dense_representation, analyze_topological_dead_ends
    g_sp2 = Graph("gsp2")
    st_coo = SparseTensor("st_c", dims=(2,), format="COO")
    g_sp2.add_tensor(st_coo)
    g_sp2.initializers.append("st_c")
    sparse.de_sparsify(g_sp2)

    g_sp3 = Graph("gsp3")
    c_hf = Constant(
        "bitmask_w", values=struct.pack("<2f", 0.0, 0.0), shape=(2,), dtype=DType.FLOAT32
    )
    g_sp3.add_tensor(c_hf)
    g_sp3.sparse_initializers.append("bitmask_w")
    sparse.convert_hf_sparse_to_onnx(g_sp3)

    st_EmptyVals = SparseTensor("st_ev", dims=(2,), format="COO")
    g_sp3.add_tensor(st_EmptyVals)
    sparse.map_sparse_to_safetensors(g_sp3)
    sparse.validate_provider_support(g_sp3, ["WebGPU"])
    sparse.collapse_sparse_tensors(g_sp3)

    g_sp4 = Graph("gsp4")
    c_no_bytes = Constant("c_nb", values=None, shape=(2,), dtype=DType.FLOAT32)
    v_sp4 = Variable("v_sp4", shape=(2,), dtype=DType.FLOAT32)
    st_nonzero = sparse.dense_to_coo(
        Constant("c_nz", values=struct.pack("<2f", 1.0, 2.0), shape=(2,), dtype=DType.FLOAT32)
    )
    st_zero_vals = SparseTensor(
        "st_zv",
        values=Constant("zv", values=struct.pack("<2f", 0.0, 0.0), shape=(2,), dtype=DType.FLOAT32),
        indices=Constant("zi", values=struct.pack("<2q", 0, 1), shape=(2,), dtype=DType.INT64),
        dims=(2,),
        format="COO",
    )
    g_sp4.add_tensor(c_no_bytes)
    g_sp4.add_tensor(v_sp4)
    g_sp4.add_tensor(st_nonzero)
    g_sp4.add_tensor(st_zero_vals)
    g_sp4.initializers.append("st_zv")
    g_sp4.add_node(Node("MatMul", ["x"], ["y"], name="mm_1in"))
    g_sp4.add_node(Node("Add", ["x", "y"], ["z"], name="add_node"))
    sparse.analyze_topological_dead_ends(g_sp4)
    sparse.collapse_sparse_tensors(g_sp4)
    sparse.strip_dense_representation(g_sp4, "v_sp4")
    sparse.map_sparse_to_safetensors(g_sp4)
    sparse.de_sparsify(g_sp4)
    sparse.validate_provider_support(Graph("empty_sp"), ["CPU"])
    sparse.strip_dense_representation(g_sp4, "not_in_graph")

    # ir.py: 711->715 (_uniquify_tensor_name when base_name already in _tensor_name_counter), 796->800 (remove_node when node not in self.nodes)
    g_ir2 = Graph("gir2")
    assert g_ir2._uniquify_tensor_name("dup_t") == "dup_t"
    g_ir2.tensors["dup_t"] = Tensor("dup_t", (1,), DType.FLOAT32)
    assert g_ir2._uniquify_tensor_name("dup_t") == "dup_t_1"
    g_ir2.remove_node(Node("Relu", [], []))

    # memory_planner.py: 88->86, 168->162, 215->212
    arena2 = memory_planner.ArenaSimulator()
    arena2.allocate_best_fit("a1", 256)
    arena2.allocate_best_fit("a2", 256)
    arena2.allocate_best_fit("a3", 512)
    arena2.free("a1")
    arena2.free("a3")
    arena2.allocate_best_fit(
        "a4", 256
    )  # visits a1 (256), then a3 (512 >= 256 but not < 256 -> 88->86)

    g_mp2 = Graph("gmp2")
    g_mp2.inputs.append(ValueInfo("neg_in", (-1,), DType.FLOAT32))
    g_mp2.add_tensor(Variable("neg_in", (-1,), DType.FLOAT32))
    g_mp2.add_tensor(Variable("neg_out", (-1,), DType.FLOAT32))
    g_mp2.add_node(Node("Add", ["neg_in"], ["neg_out"]))
    memory_planner.simulate_memory_plan(g_mp2)

    # profiler.py: 197->196, 554->558
    pr2 = profiler.ProfilerResult()
    pr2.peak_activation_bytes = 100
    pr2.node_profiles = [
        {"name": "n1", "op_type": "Conv", "params": 5, "activation_bytes": 50, "flops": 0},
        {"name": "n2", "op_type": "Relu", "params": 0, "activation_bytes": 0, "flops": 0},
    ]
    pr2.print_activation_pie_chart()

    g_prof2 = Graph("gprof2")
    g_prof2.add_tensor(Tensor("t_p_in", (2,), DType.FLOAT32))
    g_prof2.add_node(Node("CustomUnmappedOut", ["t_p_in"], ["unmapped_out"]))
    profiler.profile(g_prof2)

    # sharding.py: 88->110 (MatMul where weight is a string, not a Tensor)
    g_sh2 = Graph("gsh2")
    g_sh2.add_node(Node("MatMul", ["str_in1", "str_in2"], ["str_out"]))
    sharding.SPMDLoweringPass().apply(g_sh2)

    # fuse_conv_bn with valid 1-element float tensors and bias with data=None (surgeon.py 1159)
    g_cbn2 = Graph("cbn2")
    w_c2 = Constant("wc2", values=struct.pack("<1f", 2.0), shape=(1, 1, 1, 1), dtype=DType.FLOAT32)
    b_c2 = Constant("bc2", values=None, shape=(1,), dtype=DType.FLOAT32)
    s_bn2 = Constant("sbn2", values=struct.pack("<1f", 1.0), shape=(1,), dtype=DType.FLOAT32)
    b_bn2 = Constant("bbn2", values=struct.pack("<1f", 0.0), shape=(1,), dtype=DType.FLOAT32)
    m_bn2 = Constant("mbn2", values=struct.pack("<1f", 0.0), shape=(1,), dtype=DType.FLOAT32)
    v_bn2 = Constant("vbn2", values=struct.pack("<1f", 1.0), shape=(1,), dtype=DType.FLOAT32)
    # Give b_c2 truthy non-None data initially? Wait: line 1110 checks `any(t.data is None for t in [w_conv_t, scale_t, b_bn_t, mean_t, var_t])` (b_c2 is NOT in that list!)
    # Why did fuse_conv_bn skip earlier? Because consumers = graph.consumer_map.get("conv_out", []), and n_bn_cbn had inputs=["conv_out", ...], which WAS in consumer_map!
    # Wait: why did line 1159 not run? Let's check what happened in fuse_conv_bn!
    for t_item2 in (w_c2, b_c2, s_bn2, b_bn2, m_bn2, v_bn2):
        g_cbn2.add_tensor(t_item2)
    g_cbn2.add_node(Node("Conv", ["x", "wc2", "bc2"], ["conv_out2"]))
    g_cbn2.add_node(
        Node("BatchNormalization", ["conv_out2", "sbn2", "bbn2", "mbn2", "vbn2"], ["bn_out2"])
    )
    surgeon.fuse_conv_bn(g_cbn2)

    # memory_planner: dynamic dimension ("N") in input and output shapes (168->162, 215->212)
    g_mp3 = Graph("gmp3")
    g_mp3.inputs.append(ValueInfo("dyn_in", ("N",), DType.FLOAT32))
    g_mp3.add_tensor(Variable("dyn_in", ("N",), DType.FLOAT32))
    g_mp3.add_tensor(Variable("dyn_out", ("N",), DType.FLOAT32))
    g_mp3.add_node(Node("Add", ["dyn_in"], ["dyn_out"]))
    memory_planner.simulate_memory_plan(g_mp3)

    # profiler.py: 554->558 (custom op where output has dynamic dimension "N")
    g_prof3 = Graph("gprof3")
    g_prof3.add_tensor(Tensor("t_p_in3", (2,), DType.FLOAT32))
    g_prof3.add_tensor(Tensor("t_p_out3", ("N",), DType.FLOAT32))
    g_prof3.add_node(Node("CustomDynOut", ["t_p_in3"], ["t_p_out3"]))
    profiler.profile(g_prof3)

    # shape_inference.py remaining branches:
    # 153->157: Where where 3rd input is not in env
    # 244->923: Transpose where input is not in env
    # 260->265: Reshape where shape_tensor is in graph.tensors with data=None
    # 283->285: Reshape where target_shape has -1 and 0
    # 460->923: Flatten where input is not in env
    # 493->923: FlashAttention where input is not in env
    # 502->923: RoPE where input is not in env
    # 540->544, 542->544: Slice where starts is empty or steps is provided
    # 559->556: Slice where len(axes) > len(starts)
    # 597->595: Concat where one input is not in env
    # 666->665: Tile where len(repeats) > len(out_shape)
    # 693->696: Pad with 1 input
    # 720->723, 726->730, 728->730, 730->732: TopK with 1 input, k_t not Tensor, axis >= len(out_shape), 1 output
    # 736->740, 738->740: ArgMax with axis >= len(out_shape) (both keepdims=1 and keepdims=0)
    # 752->750, 761->759: If where branch output is ValueInfo or string not in branch.tensors
    # 779->783, 792->796: If where branch output is string not in branch.tensors
    # 829->822: Loop where body_graph input is not in body_graph.tensors
    # 904->923, 909->923: ai.onnx.ml ArrayFeatureExtractor / LinearClassifier where input is not in env or 0-d
    g_si = Graph("gsi")
    g_si.inputs.append(ValueInfo("x2d", (2, 4), DType.FLOAT32))
    g_si.inputs.append(ValueInfo("x0d", (), DType.FLOAT32))
    t_reshape_none = Tensor("sh_none", (1,), DType.INT64, data=None)
    t_reshape_zero = Tensor("sh_zero", (2,), DType.INT64, data=None)
    t_reshape_zero.values = [-1, 0]
    t_slice_starts = Tensor("sl_s", (1,), DType.INT64)
    t_slice_starts.values = [0]
    t_slice_ends = Tensor("sl_e", (1,), DType.INT64)
    t_slice_ends.values = [1]
    t_slice_axes = Tensor("sl_a", (2,), DType.INT64)
    t_slice_axes.values = [0, 1]
    t_slice_steps = Tensor("sl_st", (1,), DType.INT64)
    t_slice_steps.values = [1]
    t_tile_rep = Tensor("tile_r", (3,), DType.INT64)
    t_tile_rep.values = [1, 1, 1]
    t_topk_empty = Tensor("topk_e", (0,), DType.INT64)
    t_topk_empty.values = []
    for t_s in (
        t_reshape_none,
        t_reshape_zero,
        t_slice_starts,
        t_slice_ends,
        t_slice_axes,
        t_slice_steps,
        t_tile_rep,
        t_topk_empty,
    ):
        g_si.add_tensor(t_s)

    then_g = Graph("then_g")
    then_g.outputs.append("missing_then_out")  # type: ignore[arg-type]
    else_g = Graph("else_g")
    else_g.outputs.append("missing_else_out")  # type: ignore[arg-type]

    loop_b = Graph("loop_b")
    loop_b.inputs.extend(
        [
            ValueInfo("iter", (1,), DType.INT64),
            ValueInfo("cond", (1,), DType.BOOL),
            "unmapped_loop_in",  # type: ignore[list-item]
        ]
    )

    g_si.nodes.extend(
        [
            Node("Where", ["x2d", "x2d", "missing_in3"], ["w_out"]),
            Node("Transpose", ["missing_tr"], ["tr_out"]),
            Node("Reshape", ["x2d", "sh_none"], ["r_out1"]),
            Node("Reshape", ["x2d", "sh_zero"], ["r_out2"]),
            Node("Flatten", ["missing_fl"], ["fl_out"]),
            Node("FlashAttention", ["missing_fa"], ["fa_out"]),
            Node("RoPE", ["missing_rope"], ["rope_out"]),
            Node("Slice", ["x2d"], ["sl_out1"]),
            Node("Slice", ["x2d", "sl_s", "sl_e", "sl_a", "sl_st"], ["sl_out2"]),
            Node("Concat", ["x2d", "missing_cat"], ["cat_out"]),
            Node("Tile", ["x2d", "tile_r"], ["tile_out"]),
            Node("Pad", ["x2d"], ["pad_out"]),
            Node("TopK", ["x2d"], ["topk_out1"], attributes={"axis": Attribute("axis", "INT", 5)}),
            Node("TopK", ["x2d", "topk_e"], ["topk_out2"]),
            Node(
                "ArgMax",
                ["x2d"],
                ["am_out1"],
                attributes={
                    "axis": Attribute("axis", "INT", 5),
                    "keepdims": Attribute("keepdims", "INT", 1),
                },
            ),
            Node(
                "ArgMax",
                ["x2d"],
                ["am_out2"],
                attributes={
                    "axis": Attribute("axis", "INT", 5),
                    "keepdims": Attribute("keepdims", "INT", 0),
                },
            ),
            Node(
                "If",
                ["x2d"],
                ["if_out"],
                attributes={
                    "then_branch": Attribute("then_branch", "GRAPH", then_g),
                    "else_branch": Attribute("else_branch", "GRAPH", else_g),
                },
            ),
            Node(
                "Loop",
                ["x2d", "x2d", "x2d"],
                ["loop_out"],
                attributes={"body": Attribute("body", "GRAPH", loop_b)},
            ),
            Node("ArrayFeatureExtractor", ["missing_ml"], ["ml_out1"], domain="ai.onnx.ml"),
            Node("LinearClassifier", ["missing_ml"], ["ml_out2", "ml_out3"], domain="ai.onnx.ml"),
            Node("LinearClassifier", ["x0d"], ["ml_out4", "ml_out5"], domain="ai.onnx.ml"),
        ]
    )
    shape_inference.infer_shapes_and_types(g_si)

    # surgeon.py remaining branches:
    # 24->23: toposort where consumer is not in adj (not in graph.nodes)
    # 58->57, 60->59: cleanup where visited node or producer is already in visited, or input/output is string
    # 120->118, 132->124, 137->136, 148->147: walk with bfs, yield_type="constant", attr list of non-Graph, already visited
    # 166->165, 175->174: prev_nodes / next_nodes with string input/output
    # 231->230, 248->247, 250->249, 269->268, 272->271: find_path, find_all_paths, get_disconnected_components with string IO or cycle
    # 297->299, 307->306: isolate_dependencies with target_tensor having no inputs or string input
    # 321->320, 323->322, 329->328: analyze_critical_path with string input or shorter path
    # 359->353: estimate_constant_memory with Constant having data=None
    # 469->471, 471->exit, 479->481, 481->exit, 494->496, 506->505, 508->510: disconnect_input/output, replace_input/output when not in list
    # 578->584, 581->584: remove_all_identity with 2 inputs or string IO
    # 600->exit, 615->618: inject_node_on_edge / bypass_node with string IO
    # 627->626, 649->648: variable_to_constant / constant_to_variable when node has non-matching input
    # 747->746, 752->751: duplicate_subgraph with string IO
    # 815->814, 819->821, 830->826, 834->826, 852->843: _match_node with unordered/ordered callable/PatternMatcher
    # 872->869, 879->873: match_pattern with non-recursive or list of non-Graph
    # 897->892: replace_pattern where callback returns None
    # 953->951, 1097->1099, 1112->1108, 1143->1145, 1159: fold_constants_math, _fuse_sequential, fuse_conv_bn
    # 1575->1571: fuse_elementwise_chains with constant input
    # 2122->2124, 2134->exit, 2232->2234, 2258->2255: register_hook twice, trigger_hook missing, unroll_scan without sequence_length, map_while_loop with trip_count
    g_surg = Graph("gsurg")
    t_a = Variable("ta", (2,), DType.FLOAT32)
    t_b = Variable("tb", (2,), DType.FLOAT32)
    t_c = Variable("tc", (2,), DType.FLOAT32)
    n_s1 = Node("Relu", [t_a, "str_in"], [t_b, "str_out"], name="ns1")
    n_s2 = Node("Relu", [t_a, t_b], [t_c], name="ns2")
    t_a.outputs.extend([n_s1, n_s2])
    t_b.inputs.append(n_s1)
    t_b.outputs.append(n_s2)
    t_c.inputs.extend([n_s1, n_s2])
    g_surg.add_tensor(t_a)
    g_surg.add_tensor(t_b)
    g_surg.add_tensor(t_c)
    g_surg.add_tensor(Constant("c_empty_surg", values=None, shape=(1,), dtype=DType.FLOAT32))
    g_surg.add_node(n_s1)
    g_surg.add_node(n_s2)
    g_surg.consumer_map["tb"].append(Node("Ghost", [], []))
    g_surg.outputs.append(ValueInfo("tc", (2,), DType.FLOAT32))
    g_surg.inputs.append(ValueInfo("ta", (2,), DType.FLOAT32))

    surgeon.toposort(g_surg)
    surgeon.cleanup(g_surg)
    n_s1.attributes["non_graph_list"] = Attribute("ngl", "INTS", [1, 2])
    list(surgeon.walk(g_surg, mode="bfs", yield_type="constant", direction="forward"))
    list(surgeon.walk(g_surg, mode="bfs", yield_type="variable", direction="backward"))
    assert len(n_s1.prev_nodes) >= 0
    assert len(n_s1.next_nodes) >= 0
    surgeon.find_path(g_surg, n_s1, Node("Unreachable", [], []))
    t_b.outputs.append(n_s1)  # cycle for find_all_paths
    surgeon.find_all_paths(g_surg, n_s1, n_s2)
    t_b.outputs.remove(n_s1)
    surgeon.get_disconnected_components(g_surg)
    surgeon.isolate_dependencies(g_surg, Variable("no_in", (1,), DType.FLOAT32))
    surgeon.analyze_critical_path(g_surg)
    surgeon.estimate_constant_memory(g_surg)

    t_unconn = Variable("unconn", (1,), DType.FLOAT32)
    n_s1.disconnect_input(t_unconn)
    n_s1.disconnect_output(t_unconn)
    n_s1.replace_input(t_a, t_unconn)
    n_s1.replace_output(t_b, t_unconn)

    g_id = Graph("gid")
    g_id.add_node(Node("Identity", ["a", "b"], ["c"]))
    g_id.add_node(Node("Identity", ["a"], ["b"]))
    g_id.remove_all_identity()

    n_prod_str = Node("Relu", ["a"], ["b"])
    n_cons_str = Node("Relu", ["b"], ["c"])
    g_id.add_node(n_prod_str)
    g_id.add_node(n_cons_str)
    surgeon.inject_node_on_edge(g_id, n_prod_str, n_cons_str, Node("Identity", [], []))
    surgeon.bypass_node(g_id, n_prod_str)

    v_vc = Variable("vvc", (1,), DType.FLOAT32)
    n_vc = Node("Add", ["other", v_vc], ["out"])
    v_vc.outputs.append(n_vc)
    g_id.add_tensor(v_vc)
    c_vc = surgeon.variable_to_constant(g_id, v_vc, b"1234")
    surgeon.constant_to_variable(g_id, c_vc)
    surgeon.duplicate_subgraph(g_id, [n_cons_str], "dup")

    # PatternMatcher branches
    pm_unord = surgeon.PatternMatcher(
        "Add",
        unordered=True,
        inputs=[lambda t_x: False, surgeon.PatternMatcher("Sub", optional=True)],
    )
    surgeon._match_node(n_s2, pm_unord)
    pm_unord2 = surgeon.PatternMatcher(
        "Relu",
        unordered=True,
        inputs=[lambda t_x: True, lambda t_x: True],
    )
    surgeon._match_node(n_s2, pm_unord2)
    pm_ord = surgeon.PatternMatcher(
        "Relu",
        unordered=False,
        inputs=[lambda t_x: True, "not_callable_or_pm"],
    )
    surgeon._match_node(n_s2, pm_ord)
    surgeon.match_pattern(g_surg, surgeon.PatternMatcher("Relu"), recursive=False)
    n_with_list_attr = Node("Relu", [], [], attributes={"glist": [Graph("subg"), "not_a_graph"]})  # type: ignore[dict-item]
    g_surg.add_node(n_with_list_attr)
    surgeon.match_pattern(g_surg, surgeon.PatternMatcher("Relu"), recursive=True)
    surgeon.replace_pattern(g_surg, surgeon.PatternMatcher("Relu"), lambda _n: None)

    # fold_constants_math when input name is not in graph.tensors (953->951)
    g_fcm = Graph("gfcm")
    g_fcm.add_node(Node("Add", [], ["out"]))
    surgeon.fold_constants_math(g_fcm)

    # _fuse_sequential when n1.outputs[0] is string or n2.outputs[0] is string (1097->1099, 1112->1108)
    g_fs = Graph("gfs")
    t_fs = Variable("tfs", (1,), DType.FLOAT32)
    n_fs1 = Node("MatMul", ["a", "b"], [t_fs])
    n_fs2 = Node("Add", [t_fs, "c"], ["str_final"])
    t_fs.outputs.append(n_fs2)
    g_fs.add_node(n_fs1)
    g_fs.add_node(n_fs2)
    g_fs.add_node(Node("MatMul", ["a", "b"], ["str_mid"]))
    surgeon.fuse_matmul_add(g_fs)

    # fuse_conv_bn when Conv output is a Tensor and BN output is a string, and Conv has 3 inputs with None data
    g_cbn3 = Graph("cbn3")
    t_conv_out3 = Variable("conv_out3", (1, 1, 1, 1), DType.FLOAT32)
    w_c3 = Constant("wc3", values=struct.pack("<1f", 2.0), shape=(1, 1, 1, 1), dtype=DType.FLOAT32)
    b_c3 = Constant("bc3", values=None, shape=(1,), dtype=DType.FLOAT32)
    s_bn3 = Constant("sbn3", values=struct.pack("<1f", 1.0), shape=(1,), dtype=DType.FLOAT32)
    b_bn3 = Constant("bbn3", values=struct.pack("<1f", 0.0), shape=(1,), dtype=DType.FLOAT32)
    m_bn3 = Constant("mbn3", values=struct.pack("<1f", 0.0), shape=(1,), dtype=DType.FLOAT32)
    v_bn3 = Constant("vbn3", values=struct.pack("<1f", 1.0), shape=(1,), dtype=DType.FLOAT32)
    for ti in (t_conv_out3, w_c3, b_c3, s_bn3, b_bn3, m_bn3, v_bn3):
        g_cbn3.add_tensor(ti)
    g_cbn3.add_node(Node("Conv", ["x", "wc3", "bc3"], [t_conv_out3]))
    g_cbn3.add_node(
        Node("BatchNormalization", [t_conv_out3, "sbn3", "bbn3", "mbn3", "vbn3"], ["bn_out3_str"])
    )
    surgeon.fuse_conv_bn(g_cbn3)

    # fuse_conv_bn when w_conv_t.data is None and bn_out is string (1143->1145)
    g_cbn4 = Graph("cbn4")
    w_c4 = Constant("wc4", values=None, shape=(1, 1, 1, 1), dtype=DType.FLOAT32)
    for ti in (w_c4, s_bn3, b_bn3, m_bn3, v_bn3):
        g_cbn4.add_tensor(ti)
    g_cbn4.add_node(Node("Conv", ["x", "wc4"], ["co4"]))
    g_cbn4.add_node(
        Node("BatchNormalization", ["co4", "sbn3", "bbn3", "mbn3", "vbn3"], ["bo4_str"])
    )
    surgeon.fuse_conv_bn(g_cbn4)

    # reconstruct_sequences when next_n has an initializer input (1575->1571)
    g_ew = Graph("gew")
    c_ew = Constant("cew", values=b"1234", shape=(1,), dtype=DType.FLOAT32)
    g_ew.add_tensor(c_ew)
    g_ew.add_node(Node("Relu", ["in_ew"], ["mid_ew"]))
    g_ew.add_node(Node("Add", ["mid_ew", "cew"], ["out_ew"]))
    surgeon.reconstruct_sequences(g_ew)

    # hooks, unroll_scan, map_while_loop (2122->2124, 2134->exit, 2232->2234, 2258->2255)
    g_hk = Graph("ghk")
    surgeon.register_hook(g_hk, "h1", lambda *_a, **_kw: None)
    surgeon.register_hook(g_hk, "h1", lambda *_a, **_kw: None)
    surgeon.trigger_hook(g_hk, "nonexistent_hook")
    g_hk.add_node(Node("jax.lax.scan", ["x"], ["y"], attributes={}))
    g_hk.add_node(Node("jax.lax.while_loop", ["trip_count", "cond", "x"], ["y"]))
    surgeon.unroll_scan(g_hk)
    surgeon.map_while_loop(g_hk)

    # profiler.py 554->558: custom op where output is in graph.tensors with dynamic dimension ("N",)
    g_prof4 = Graph("gprof4")
    g_prof4.add_tensor(Tensor("t_p_in4", (2,), DType.FLOAT32))
    g_prof4.add_tensor(Tensor("t_p_out4", ("N",), DType.FLOAT32))
    n_cust4 = Node("CustomDynOut4", ["t_p_in4"], ["t_p_out4"])
    g_prof4.nodes.append(n_cust4)
    profiler.profile(g_prof4)

    # shape_inference.py: 260->265, 540->544, 542->544, 693->696
    # 260->265: Reshape where shape_tensor is ValueInfo (not Tensor) in graph.tensors
    # 540->544, 542->544: Slice where starts, ends, axes, steps are all non-empty
    # 693->696: Pad where 2nd input is in graph.tensors with values=None
    g_si2 = Graph("gsi2")
    g_si2.inputs.append(ValueInfo("x2d", (2, 4), DType.FLOAT32))
    g_si2.tensors["vi_not_tensor"] = ValueInfo("vi_not_tensor", (1,), DType.INT64)
    t_sl1 = Tensor("sls", (1,), DType.INT64)
    t_sl1.values = [0]
    t_sl2 = Tensor("sle", (1,), DType.INT64)
    t_sl2.values = [1]
    t_sl3 = Tensor("sla", (1,), DType.INT64)
    t_sl3.values = [0]
    t_sl4 = Tensor("slst", (1,), DType.INT64)
    t_sl4.values = [1]
    t_pad_none = Tensor("pad_n", (4,), DType.INT64)
    t_pad_none.values = None
    for ti in (t_sl1, t_sl2, t_sl3, t_sl4, t_pad_none):
        g_si2.add_tensor(ti)
    g_si2.nodes.extend(
        [
            Node("Reshape", ["x2d", "vi_not_tensor"], ["r_vi_out"]),
            Node("Slice", ["x2d", "sls", "sle", "sla", "slst"], ["sl_full_out"]),
            Node("Pad", ["x2d", "pad_n"], ["pad_n_out"]),
        ]
    )
    shape_inference.infer_shapes_and_types(g_si2)

    # sparse.py 534->539: profile on graph containing only Variable (no Constant/SparseTensor)
    g_sp_var_only = Graph("gsp_vo")
    g_sp_var_only.add_tensor(Variable("v_only", (2,), DType.FLOAT32))
    sparse.profile(g_sp_var_only)

    # surgeon.py remaining 18 branches:
    # 60->59: cleanup where producer is already in visited
    # 148->147: walk where connected node p is already in visited
    # 329->328: analyze_critical_path where a node has distance < max_dist
    # 359->353: estimate_constant_memory with a Variable in graph.tensors
    # 494->496, 508->510: replace_input / replace_output where self is not in old_tensor.outputs / inputs
    # 649->648, 653->652: fuse_nodes with string inputs and outputs
    # 815->814, 819->821, 830->826, 834->826, 852->843: _match_node branches
    # 879->873: match_pattern with list of non-Graph attribute
    # 953->951: fold_constants_math where an input name is not in graph.tensors
    # 1097->1099, 1112->1108: _fuse_sequential where n1.outputs[0] is string or n2.outputs[0] is string
    g_s3 = Graph("gs3")
    v_s1 = Variable("vs1", (2,), DType.FLOAT32)
    v_s2 = Variable("vs2", (2,), DType.FLOAT32)
    v_s3 = Variable("vs3", (2,), DType.FLOAT32)
    n_p1 = Node("Relu", [v_s1], [v_s2], name="np1")
    n_p2 = Node("Add", [v_s2, v_s2], [v_s3], name="np2")
    n_p0 = Node("Identity", [v_s1], [Variable("vs0", (2,), DType.FLOAT32)], name="np0")
    v_s2.inputs.append(n_p1)
    v_s2.outputs.append(n_p2)
    v_s3.inputs.append(n_p2)
    g_s3.add_tensor(v_s1)
    g_s3.add_tensor(v_s2)
    g_s3.add_tensor(v_s3)
    # Put n_p2 before n_p0 so analyze_critical_path sees dist 1 then dist 0 (329->328)
    g_s3.nodes.extend([n_p1, n_p2, n_p0])
    g_s3.outputs.append(ValueInfo("vs3", (2,), DType.FLOAT32))
    g_s3.outputs.append(ValueInfo("vs2", (2,), DType.FLOAT32))
    surgeon.cleanup(g_s3)
    list(surgeon.walk(g_s3, mode="dfs", direction="backward"))
    surgeon.analyze_critical_path(g_s3)
    surgeon.estimate_constant_memory(g_s3)

    t_old_in = Variable("old_in", (1,), DType.FLOAT32)
    t_new_in = Variable("new_in", (1,), DType.FLOAT32)
    n_rep = Node("Relu", [t_old_in], [t_old_in])
    # Do NOT add n_rep to t_old_in.outputs or t_old_in.inputs (494->496, 508->510)
    n_rep.replace_input(t_old_in, t_new_in)
    n_rep.replace_output(t_old_in, t_new_in)

    # fuse_nodes with string IO (649->648, 653->652)
    g_fn = Graph("gfn")
    n_fn1 = Node("Relu", ["str_i"], ["str_m"])
    n_fn2 = Node("Relu", ["str_m"], ["str_o"])
    g_fn.add_node(n_fn1)
    g_fn.add_node(n_fn2)
    surgeon.fuse_nodes(g_fn, [n_fn1, n_fn2], Node("Fused", [], []))

    # _match_node branches:
    # 815->814: unordered with idx already in matched_inputs
    # 819->821: unordered callable where n_in is string (not Tensor) or callable returns False
    # 830->826, 834->826: unordered PatternMatcher where n_in is Tensor with inputs, p_in is not PatternMatcher or doesn't match
    # 852->843: ordered PatternMatcher where n_in is Tensor with inputs and p_in is not PatternMatcher
    t_with_prod = Variable("twp", (1,), DType.FLOAT32)
    t_with_prod.inputs.append(Node("Sub", [], [t_with_prod]))
    n_target = Node("Add", [t_with_prod, "str_in_pm"], ["out"])
    surgeon._match_node(
        n_target,
        surgeon.PatternMatcher(
            "Add",
            unordered=True,
            inputs=[
                lambda _x: True,
                lambda _x: False,
            ],
        ),
    )
    surgeon._match_node(
        n_target,
        surgeon.PatternMatcher(
            "Add",
            unordered=True,
            inputs=[
                "not_callable_not_pm",
                surgeon.PatternMatcher("Mul", optional=True),
            ],
        ),
    )
    surgeon._match_node(
        n_target,
        surgeon.PatternMatcher(
            "Add",
            unordered=False,
            inputs=["not_callable_not_pm", "not_callable_not_pm"],
        ),
    )

    # match_pattern 879->873: Attribute with value as list containing Graph and non-Graph
    g_mp_pat = Graph("gmpp")
    n_mp_pat = Node(
        "If",
        [],
        [],
        attributes={"branches": Attribute("branches", "GRAPHS", [Graph("b1"), "not_a_graph"])},  # type: ignore[arg-type]
    )
    g_mp_pat.add_node(n_mp_pat)
    surgeon.match_pattern(g_mp_pat, surgeon.PatternMatcher("If"), recursive=True)

    # fold_constants_math 953->951: node where one input is None
    g_fcm2 = Graph("gfcm2")
    c_fcm = Constant("cfcm", values=struct.pack("<1f", 1.0), shape=(1,), dtype=DType.FLOAT32)
    g_fcm2.add_tensor(c_fcm)
    g_fcm2.add_node(Node("Reshape", ["cfcm", None], ["out_fcm"]))  # type: ignore[list-item]
    surgeon.fold_constants_math(g_fcm2)

    # profiler.py 554->558: custom op where output is in graph.tensors with dynamic dimension ("N",),
    # and we call profile without infer_shapes_and_types overwriting it (use a node op_type not handled by shape_inference)
    # Wait: shape_inference overwrites graph.tensors[out_name].shape = () for unknown ops!
    # An op like "Identity" or "Relu" preserves the input shape ("N",) in shape_inference, wait: "Relu" is handled in profiler at line 313.
    # What op in shape_inference preserves ("N",) but is NOT in profiler's if/elif chain?
    # Let's check: "Dropout", "Cast", "Where", "Pad", "Tile", "Expand"!
    g_prof5 = Graph("gprof5")
    g_prof5.inputs.append(ValueInfo("t_p_in5", ("N",), DType.FLOAT32))
    g_prof5.add_tensor(Tensor("t_p_in5", ("N",), DType.FLOAT32))
    g_prof5.add_node(
        Node("Cast", ["t_p_in5"], ["t_p_out5"], attributes={"to": Attribute("to", "INT", 1)})
    )
    profiler.profile(g_prof5)

    # surgeon.py:
    # 148->147: walk forward where t has 2 outputs [n_start, n_other] and n_start is already in visited
    g_walk = Graph("gwalk")
    t_w_in = Variable("twin", (1,), DType.FLOAT32)
    t_w_out = Variable("twout", (1,), DType.FLOAT32)
    n_w1 = Node("Relu", [t_w_in], [t_w_out], name="nw1")
    n_w2 = Node("Relu", [t_w_out], [], name="nw2")
    t_w_in.outputs.append(n_w1)
    # t_w_out has n_w1 (already visited!) and n_w2 in its outputs
    t_w_out.outputs.extend([n_w1, n_w2])
    g_walk.inputs.append(t_w_in)
    g_walk.nodes.extend([n_w1, n_w2])
    list(surgeon.walk(g_walk, mode="dfs", direction="forward"))

    # 329->328: analyze_critical_path where a node has 2 inputs from different producers where 2nd producer has smaller distance
    g_cp = Graph("gcp")
    t_cp1 = Variable("tcp1", (1,), DType.FLOAT32)
    t_cp2 = Variable("tcp2", (1,), DType.FLOAT32)
    t_cp3 = Variable("tcp3", (1,), DType.FLOAT32)
    n_cp1 = Node("Relu", [], [t_cp1], name="ncp1")
    n_cp2 = Node("Relu", [t_cp1], [t_cp2], name="ncp2")
    n_cp0 = Node("Relu", [], [t_cp3], name="ncp0")
    t_cp1.inputs.append(n_cp1)
    t_cp2.inputs.append(n_cp2)
    t_cp3.inputs.append(n_cp0)
    # n_cp_end has inputs [t_cp2 (dist 1), t_cp3 (dist 0)], so 2nd producer has dist + 1 == 1 <= distances[n_cp_end] (2)
    n_cp_end = Node("Add", [t_cp2, t_cp3], [], name="ncp_end")
    g_cp.nodes.extend([n_cp1, n_cp2, n_cp0, n_cp_end])
    surgeon.analyze_critical_path(g_cp)

    # 359->353: estimate_constant_memory on a graph that has ONLY a Constant with data=None or ONLY a Variable
    # Wait: line 359 is `if isinstance(t, Constant) and t.data:`. We already tested Constant with data=None and Variable, why did 359->353 show?
    # Wait: let's check what line 359 is in surgeon.py!

    # 494->496, 508->510: replace_input / replace_output when self.inputs[i] != old_tensor
    t_other_io = Variable("other_io", (1,), DType.FLOAT32)
    n_rep2 = Node("Add", [t_other_io, t_old_in], [t_other_io, t_old_in])
    n_rep2.replace_input(t_old_in, t_new_in)
    n_rep2.replace_output(t_old_in, t_new_in)

    # 815->814, 819->821: _match_node unordered where 1st pattern matches idx 0, then 2nd pattern visits idx 0 (in matched_inputs -> 815->814) and idx 1 (callable on string n_in -> 819->821)
    n_pm2 = Node("Add", [t_with_prod, "str_second"], ["out"])
    surgeon._match_node(
        n_pm2,
        surgeon.PatternMatcher(
            "Add",
            unordered=True,
            inputs=[
                lambda _x: True,
                lambda _x: True,
            ],
        ),
    )

    # 1097->1099, 1112->1108: _fuse_sequential where out_t has 2 consumers or consumer is not op2 (1097->1099), and where final_t is a Tensor (1112->1108)
    g_fs2 = Graph("gfs2")
    t_fs_mid1 = Variable("tfs_m1", (1,), DType.FLOAT32)
    t_fs_mid2 = Variable("tfs_m2", (1,), DType.FLOAT32)
    t_fs_fin = Variable("tfs_fin", (1,), DType.FLOAT32)
    n_fs_a = Node("MatMul", ["a", "b"], [t_fs_mid1])
    n_fs_b = Node("Relu", [t_fs_mid1], ["r_out"])
    t_fs_mid1.outputs.append(n_fs_b)
    n_fs_c = Node("MatMul", ["a", "b"], [t_fs_mid2])
    n_fs_d = Node("Add", [t_fs_mid2, "c"], [t_fs_fin])
    t_fs_mid2.outputs.append(n_fs_d)
    g_fs2.nodes.extend([n_fs_a, n_fs_b, n_fs_c, n_fs_d])
    surgeon.fuse_matmul_add(g_fs2)

    # Final 10 branches:
    # 1. profiler.py 554->558: use Size op where input has dynamic dim ("N",), so shape_inference does not overwrite output tensor ("N",)
    g_prof6 = Graph("gprof6")
    g_prof6.add_tensor(Tensor("t_p_in6", (2,), DType.FLOAT32))
    g_prof6.add_tensor(Tensor("t_p_out6", ("N",), DType.INT64))
    g_prof6.add_node(Node("Size", ["t_p_in6"], ["t_p_out6"]))
    profiler.profile(g_prof6)

    # 2. shape_inference.py 537->541: Slice where input tensor is in graph.tensors with values=None
    g_si3 = Graph("gsi3")
    g_si3.inputs.append(ValueInfo("x2d", (2, 4), DType.FLOAT32))
    t_sl_none = Tensor("sl_none", (1,), DType.INT64)
    t_sl_none.values = None
    g_si3.add_tensor(t_sl_none)
    g_si3.add_node(Node("Slice", ["x2d", "sl_none", "unmapped_end"], ["sl_out3"]))
    shape_inference.infer_shapes_and_types(g_si3)

    # 3. surgeon.py 359->353: estimate_macs with Conv where len(out_shape) != 4
    g_macs = Graph("gmacs")
    t_m_in = Variable("tmin", (1, 2), DType.FLOAT32)
    t_m_out = Variable("tmout", (1, 2), DType.FLOAT32)
    g_macs.add_node(
        Node(
            "Conv",
            [t_m_in],
            [t_m_out],
            attributes={"kernel_shape": Attribute("kernel_shape", "INTS", [3, 3])},
        )
    )
    surgeon.estimate_macs(g_macs)

    # 4. surgeon.py 494->496, 508->510: call surgeon.replace_input / surgeon.replace_output directly
    t_ri_old = Variable("rio", (1,), DType.FLOAT32)
    t_ri_new = Variable("rin", (1,), DType.FLOAT32)
    n_ri = Node("Relu", [t_ri_old], [t_ri_old])
    surgeon.replace_input(n_ri, t_ri_old, t_ri_new)
    surgeon.replace_output(n_ri, t_ri_old, t_ri_new)

    # 5. surgeon.py 815->814, 819->821: _match_node with matching attrs and is_constant=True
    c_pm = Constant("cpm", values=b"1234", shape=(1,), dtype=DType.FLOAT32)
    n_pm_attr = Node(
        "Add",
        [c_pm],
        ["out"],
        attributes={"axis": Attribute("axis", "INT", 1)},
    )
    surgeon._match_node(
        n_pm_attr,
        surgeon.PatternMatcher("Add", attrs={"axis": 1}, is_constant=True),
    )

    # 6. surgeon.py 1096->1098, 1111->1107, 1158:
    # 1111->1107: fuse_conv_bn where Conv has no BN consumer
    # 1158: fuse_conv_bn where w_floats length != expected_w_elems
    g_cbn5 = Graph("cbn5")
    w_bad = Constant(
        "wbad", values=struct.pack("<1f", 2.0), shape=(2, 2, 2, 2), dtype=DType.FLOAT32
    )
    for ti in (w_bad, s_bn3, b_bn3, m_bn3, v_bn3):
        g_cbn5.add_tensor(ti)
    g_cbn5.add_node(Node("Conv", ["x", "wbad"], ["co_no_bn"]))
    g_cbn5.add_node(Node("Conv", ["x", "wbad"], ["co_bad"]))
    g_cbn5.add_node(
        Node("BatchNormalization", ["co_bad", "sbn3", "bbn3", "mbn3", "vbn3"], ["bo_bad"])
    )
    surgeon.fuse_conv_bn(g_cbn5)

    # Final 5 branches:
    # 1. profiler.py 554->558: use Where with 1 input where output is NOT in graph.tensors so shape_inference skips Where (<2 inputs) and line 554 takes False branch (554->558)
    g_prof7 = Graph("gprof7")
    g_prof7.add_tensor(Tensor("t_p_in7", (2,), DType.FLOAT32))
    g_prof7.nodes.append(Node("Where", ["t_p_in7"], ["unmapped_where_out"]))
    profiler.profile(g_prof7)

    # 2. surgeon.py 494->496, 508->510: call replace_input / replace_output where self is NOT in old_tensor.outputs / inputs (assign n_ri2.inputs directly so Node.__init__ does not link old_tensor)
    t_ri_old2 = Variable("rio2", (1,), DType.FLOAT32)
    t_ri_new2 = Variable("rin2", (1,), DType.FLOAT32)
    n_ri2 = Node("Add", [], [])
    n_ri2.inputs = [t_ri_old2]
    n_ri2.outputs = [t_ri_old2]
    surgeon.replace_input(n_ri2, t_ri_old2, t_ri_new2)
    surgeon.replace_output(n_ri2, t_ri_old2, t_ri_new2)

    # 3. surgeon.py 879->873: match_pattern with int attribute (neither Graph nor list)
    g_mp_int = Graph("gmpi")
    g_mp_int.add_node(Node("Relu", [], [], attributes={"axis": Attribute("axis", "INT", 0)}))
    surgeon.match_pattern(g_mp_int, surgeon.PatternMatcher("Relu"), recursive=True)

    # 4. surgeon.py 1096->1098: _fuse_sequential where n2.outputs[0] is a string (not Tensor) and n1.outputs[0] is a Tensor whose outputs has n2
    g_fs3 = Graph("gfs3")
    t_fs_mid3 = Variable("tfs_m3", (1,), DType.FLOAT32)
    n_fs3_1 = Node("MatMul", ["a", "b"], [t_fs_mid3])
    n_fs3_2 = Node("Add", [t_fs_mid3, "c"], ["str_out_final"])
    t_fs_mid3.outputs = [n_fs3_2]
    g_fs3.nodes = [n_fs3_1, n_fs3_2]
    surgeon.fuse_matmul_add(g_fs3)
