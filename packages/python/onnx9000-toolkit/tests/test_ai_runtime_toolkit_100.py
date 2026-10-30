"""Exhaustive 100% coverage tests for Python AI/Runtime/Toolkit packages (Phase 6.3.6)."""

import ast
import mmap
import struct
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from onnx9000.core import onnx_pb2 as pb
from onnx9000.core.dtypes import DType
from onnx9000.core.exceptions import CompilationError
from onnx9000.core.ir import Attribute, Graph, Node, Tensor
from onnx9000.jax.importer import JaxprImporter
from onnx9000.jax.jaxpr_string_parser import parse_jaxpr_string
from onnx9000.toolkit.gguf.parser import GGUFParser
from onnx9000.toolkit.numerical_debugger import NumericalDebugger
from onnx9000.toolkit.safetensors.parser import (
    SafeTensors,
    SafeTensorsSharded,
    save,
    save_sharded,
)
from onnx9000.toolkit.script import op
from onnx9000.toolkit.script.builder import GraphBuilder
from onnx9000.toolkit.script.parser import ScriptParser, script
from onnx9000.toolkit.script.schema import OpSchema, registry, set_target_opset, validate_op
from onnx9000.toolkit.script.var import Var
from onnx9000.toolkit.training.autograd.compiler import (
    AOTBuilder,
    apply_peft_config,
    build_backward_graph,
    cast_gradients_to_fp32,
    enforce_webgpu_limits,
    ensure_no_microsoft_opsets,
    inject_bitfit,
    inject_custom_loss_subgraph,
    load_training_checkpoint,
    optimize_intermediate_casts,
    save_training_checkpoint,
    scale_backward_graph_for_mixed_precision,
    set_eval_mode,
)
from onnx9000.toolkit.training.autograd.memory import optimize_backward_memory
from onnx9000.toolkit.training.autograd.optimizers import (
    add_adadelta_optimizer,
    add_adagrad_optimizer,
    add_adam_optimizer,
    add_adamw_optimizer,
    add_differential_privacy_noise,
    add_rmsprop_optimizer,
)
from onnx9000.toolkit.training.autograd.rules import get_vjp_rule
from onnx9000.toolkit.training.autograd.shape_inference import infer_backward_shapes
from onnx9000.toolkit.training.autograd.utils import (
    add_synchronous_barrier,
    compile_multi_replica_graph,
    embed_distributed_identifiers,
)
from onnx9000.zoo.sync import HfApi


def test_jax_and_zoo_remaining_branches() -> None:
    """Cover remaining statements and branches in onnx9000-jax and onnx9000-zoo."""
    assert HfApi().list_models() == []

    imp = JaxprImporter()
    g = imp.parse(
        {
            "invars": [],
            "constvars": [],
            "eqns": [],
            "outvars": [{"name": "missing_out", "shape": [], "type": "f32"}],
        }
    )
    assert len(g.outputs) == 0

    jaxpr_text = """
{ lambda ; a:f32[]. let
    b:f32[] = custom_op[ , flag, name='val"inner', tup=(1, 2), bad=unquoted_ident ] a
  in (b,)
}
"""
    parsed = parse_jaxpr_string(jaxpr_text)
    assert len(parsed["eqns"]) == 1  # type: ignore[arg-type]
    assert parsed["eqns"][0]["params"]["name"] == 'val"inner'  # type: ignore[index]
    assert parsed["eqns"][0]["params"]["bad"] == "unquoted_ident"  # type: ignore[index]

    parsed_no_attr = parse_jaxpr_string("b:f32[] = add[] a")
    assert parsed_no_attr["eqns"][0]["params"] == {}  # type: ignore[index]


def test_numerical_debugger_and_gguf_parser(tmp_path: Path) -> None:
    """Cover NumericalDebugger and GGUFParser remaining lines and branches."""
    g = Graph("dbg_g")
    dbg = NumericalDebugger(g)
    ep1 = MagicMock()
    ep2 = MagicMock()
    ep1.execute.return_value = {
        "y": Tensor("y", [2], data=np.array([1.0, 2.0], dtype=np.float32)),
        "non_np": Tensor("non_np", [1], data=b"raw"),
        "only_ep1": Tensor("only_ep1", [1], data=np.array([1.0], dtype=np.float32)),
    }
    ep2.execute.return_value = {
        "y": Tensor("y", [2], data=np.array([1.5, 2.5], dtype=np.float32)),
        "non_np": Tensor("non_np", [1], data=b"raw"),
    }
    errors = dbg.compare({"x": np.array([1.0, 2.0], dtype=np.float32)}, ep1, ep2)
    assert errors["y"] == pytest.approx(0.5)

    # GGUFParser: keys(), __contains__, and repeated __exit__ when view/mm/fd are None
    gguf_file = tmp_path / "empty.gguf"
    gguf_file.write_bytes(struct.pack("<4sIQQ", b"GGUF", 3, 0, 0))
    with GGUFParser(gguf_file) as parser:
        assert parser.keys() == []
        assert ("non_existent" in parser) is False
    parser.__exit__(None, None, None)


def test_safetensors_parser_remaining_branches(tmp_path: Path) -> None:
    """Cover remaining partial branches in safetensors/parser.py."""
    st_bytes = save(
        {
            "t_bool": np.array([True, False], dtype=np.bool_),
            "t_other_dt": SimpleNamespace(
                tobytes=lambda: b"\x00\x01",
                shape=(2,),
                dtype="COMPLEX64",
            ),
            "t_no_shape_dtype": SimpleNamespace(tobytes=lambda: b"\x00\x01"),
            "t_proto_unknown_dt": SimpleNamespace(raw_data=b"\x00\x01", dims=[2], data_type=999),
        },
        metadata={"format": "custom", "version": "2.0"},
    )
    st_path = tmp_path / "model.safetensors"
    st_path.write_bytes(st_bytes)

    with patch.object(mmap, "MADV_WILLNEED", 3, create=True):
        with SafeTensors(str(st_path), mmap_hint=False) as st:
            # Tensor with empty shape [] so `if dtype in struct_char_map and shape:` is False
            st.tensors["scalar_u8"] = {"dtype": "U8", "shape": [], "data_offsets": [0, 1]}
            assert len(st.get_tensor("scalar_u8")) == 1

            # get_pinned_tensor when libc has no mlock, and on win32 when VirtualLock succeeds (res != 0)
            with patch("ctypes.CDLL", return_value=SimpleNamespace()):
                mv = st.get_pinned_tensor("t_bool")
                mv.release()
            with (
                patch("sys.platform", "win32"),
                patch(
                    "ctypes.windll",
                    SimpleNamespace(kernel32=SimpleNamespace(VirtualLock=lambda *_: 1)),
                    create=True,
                ),
            ):
                mv2 = st.get_pinned_tensor("t_bool")
                mv2.release()

    # SafeTensorsSharded: call _get_file twice for the same shard so `filename not in self._files` is False
    save_sharded({"a": b"\x01\x02"}, str(tmp_path), max_shard_size=1024)
    idx_path = str(tmp_path / "model.safetensors.index.json")
    with SafeTensorsSharded(idx_path) as sharded:
        assert len(sharded.get_tensor("a")) == 2
        assert len(sharded.get_tensor("a")) == 2

    # save_sharded with empty tensors dict so `if current_shard_tensors:` is False
    empty_dir = tmp_path / "empty_sharded"
    empty_dir.mkdir()
    save_sharded({}, str(empty_dir))


def test_script_schema_builder_and_parser_remaining() -> None:
    """Cover remaining statements and branches in script/schema.py, builder.py, and parser.py."""
    # schema.py: register two versions of FutureOp, get_schema when candidates is empty, and validate_op when op_type not in registry.schemas
    registry.register(OpSchema("FutureOp", 99, ["X"], ["Y"], []))
    registry.register(OpSchema("FutureOp", 100, ["X"], ["Y"], []))
    set_target_opset(18)
    assert registry.get_schema("FutureOp", 10) is None
    validate_op("UnregisteredCustomOp", [], {})

    # builder.py
    b = GraphBuilder("b_full")
    b.set_metadata(doc_string="doc", domain="ai.custom", version=3)
    b.metadata["custom_domain"] = "com.custom"
    v_f32 = b.add_input("x_f32", DType.FLOAT32, (2, "N"))
    b.add_input("x_i64", DType.INT64, (2,))
    b.add_input("x_i16", DType.INT16, (2,))
    b.add_input("unused_in", DType.FLOAT32, (1,))
    b.add_initializer("w_f32", np.array([1.0, 2.0], dtype=np.float32))
    b.add_initializer("w_i64", np.array([1, 2], dtype=np.int64))
    b.add_initializer("w_i32", np.array([1, 2], dtype=np.int32))
    b.add_initializer("unused_init", np.array([1.0], dtype=np.float32))

    n_add = Node("Add", ["x_f32", "w_f32"], ["sum_out", "sum_out2"], {}, "add_node")
    b.add_node(n_add)
    assert b.get_node("add_node") is n_add

    # delete when node not in nodes, and when node.name not in node_by_name
    orphan = Node("Relu", ["x_f32"], ["o"], {}, "orphan")
    b.delete(orphan)
    b.add_node(orphan)
    del b.node_by_name["orphan"]
    b.delete(orphan)

    # replace when old_node not in nodes, and when old_node.name not in node_by_name and new_node.name is "" vs non-empty
    b.replace(orphan, Node("Relu", ["x_f32"], ["o"], {}, "r_new"))
    n_to_rep = Node("Relu", ["x_f32"], ["o1"], {}, "rep_me")
    b.add_node(n_to_rep)
    del b.node_by_name["rep_me"]
    n_unnamed = Node("Sigmoid", ["x_f32"], ["o1"], {}, "")
    b.replace(n_to_rep, n_unnamed)
    n_named = Node("Sigmoid", ["x_f32"], ["o1"], {}, "named_sig")
    b.replace(n_unnamed, n_named)
    b.delete(n_named)

    # replace_input when node not in self.nodes
    b.replace_input(orphan, Var("x_f32"), Var("x_i64"))

    # rename_all when a node has empty name "" (on a separate builder so b's nodes are not mutated)
    b_copy = GraphBuilder("copy")
    n_empty_name = Node("Relu", ["x_f32"], ["empty_out"], {}, "temp")
    b_copy.add_node(n_empty_name)
    n_empty_name.name = ""
    b_copy.rename_all("pfx")

    # infer_shapes edge cases: Constant (f32, i64, no value), If (with and without then_branch), Add (with missing input), CustomOp
    sub_b = GraphBuilder("sub_b")
    sub_b.add_input("sub_in", DType.FLOAT32, (2,))
    sub_b.add_output(Var("sub_in"))
    b.add_node(
        Node(
            "Constant",
            [],
            ["c_f32"],
            {"value": np.array([1.0], dtype=np.float32), "sub": sub_b},
            "c1",
        )
    )
    b.add_node(Node("Constant", [], ["c_i64"], {"value": np.array([1], dtype=np.int64)}, "c2"))
    b.add_node(Node("Constant", [], ["c_noval"], {}, "c3"))
    b.add_node(Node("Add", ["missing_in"], ["add_miss"], {}, "add_m"))
    b.add_node(Node("If", ["c_f32"], ["if_out1"], {"then_branch": sub_b}, "if1"))
    b.add_node(Node("If", ["c_f32"], ["if_out2"], {}, "if2"))
    b.add_node(Node("CustomOp", ["c_f32"], ["cust_out"], {}, "cust1"))
    b.infer_shapes()

    # extract_subgraph: diamond dependency, shared multi-output producer, direct GraphBuilder attribute (assigned after Node init), non-GraphBuilder attribute
    n_mid = Node("Add", ["sum_out", "x_f32"], ["mid_out"], {}, "mid_add")
    b.add_node(n_mid)
    n_final = Node(
        "Add",
        ["mid_out", "sum_out2"],
        ["final_y"],
        {"sub_wrapped": Attribute("sub_wrapped", "GRAPH", sub_b), "alpha": 1.0},
        "final_add",
    )
    n_final.attributes["sub_direct"] = sub_b  # type: ignore[assignment]
    b.add_node(n_final)
    b.add_output(Var("final_y"))
    b.add_output(Var("unused_out"))
    extracted = b.extract_subgraph(["x_f32", "non_existent_in"], ["final_y", "non_existent_out"])
    assert len(extracted.nodes) == 3

    # build()
    ir_g = b.build()
    assert len(ir_g.nodes) > 0

    # to_onnx and from_onnx with all attribute types
    b_onnx = GraphBuilder("onnx_test")
    b_onnx.set_metadata(doc_string="d", domain="dom", version=2)
    b_onnx.metadata["custom_domain"] = "cdom"
    b_onnx.add_input("in_f", DType.FLOAT32, (2, "B"))
    b_onnx.add_input("in_i", DType.INT64, (2,))
    b_onnx.add_input("in_other", DType.INT16, (2,))
    b_onnx.add_initializer("init_f", np.array([1.0, 2.0], dtype=np.float32))
    b_onnx.add_initializer("init_i64", np.array([1, 2], dtype=np.int64))
    b_onnx.add_initializer("init_i32", np.array([1, 2], dtype=np.int32))
    b_onnx.add_output(Var("out_y"))
    n_all_attrs = Node(
        "Custom",
        ["in_f"],
        ["out_y"],
        {
            "wrapped": Attribute("wrapped", "FLOAT", 1.5),
            "f": 2.5,
            "i": 3,
            "s": "hello",
            "t_f32": np.array([1.0], dtype=np.float32),
            "t_i64": np.array([1], dtype=np.int64),
            "ints": [1, 2],
            "floats": [1.5, 2.5],
            "strs": ["a", "b"],
            "g": sub_b,
            "raw": b"bytes",
        },
        "custom_n",
    )
    n_all_attrs.attributes["t_i32"] = np.array([1], dtype=np.int32)  # type: ignore[assignment]
    n_no_name = Node("Relu", ["in_f"], ["r_out"], {}, "tmp")
    b_onnx.add_node(n_all_attrs)
    b_onnx.add_node(n_no_name)
    n_no_name.name = ""
    proto = b_onnx.to_onnx()
    proto.graph.initializer.pop()
    # Remove the t_i32 attribute (which has empty raw_data) and add scalar tensor attr (empty dims) and unknown attr type (8)
    proto.graph.node[0].attribute.pop()
    scalar_attr = pb.AttributeProto()
    scalar_attr.name = "t_scalar"
    scalar_attr.type = 4
    scalar_attr.t.data_type = 1
    scalar_attr.t.raw_data = np.array(1.0, dtype=np.float32).tobytes()
    unk_attr = pb.AttributeProto()
    unk_attr.name = "unk"
    unk_attr.type = 8
    proto.graph.node[0].attribute.append(scalar_attr)
    proto.graph.node[0].attribute.append(unk_attr)
    reloaded = GraphBuilder.from_onnx(proto)
    assert len(reloaded.nodes) == 2

    # validate() with diamond dependency so `if n.name in visited: return` is hit
    b_val = GraphBuilder("val")
    b_val.add_node(Node("Relu", ["x"], ["a"], {}, "n1"))
    b_val.add_node(Node("Relu", ["a"], ["b1"], {}, "n2"))
    b_val.add_node(Node("Relu", ["a"], ["b2"], {}, "n3"))
    b_val.add_node(Node("Add", ["b1", "b2"], ["c"], {}, "n4"))
    b_val.validate()
    assert v_f32.name == "x_f32"

    # parser.py remaining branches:
    # - Assign with multiple targets (`a = b = x`), non-Name target (`obj.attr = x`), Tuple target with non-tuple value, Tuple target with non-Name element
    # - Call via attribute (`ns.helper(x)`), inline @script function with multiple outputs
    # - For loop with tuple target (`for i, j in x:`) and >1 carried variables
    # - While loop with 0 carried variables and >1 carried variables
    # - Return with no value (`return`)
    @script
    def multi_out_sub(u: Var) -> tuple[Var, Var]:
        """Subgraph returning two outputs."""
        return u + 1, u - 1

    ns = SimpleNamespace(multi_out_sub=multi_out_sub)
    sp = ScriptParser({"op": op, "multi_out_sub": multi_out_sub, "ns": ns})

    def complex_ast_fn(x: Var) -> None:
        """Exercise remaining AST branches in ScriptParser."""
        a = _b = x
        a.attr = x  # type: ignore[attr-defined]
        m = multi_out_sub(x)
        _c, _d = x  # non-tuple assigned to Tuple target
        (e, e.attr) = m
        p, q = ns.multi_out_sub(x)
        for _k in x:
            op.Relu(x)
        for _i, _j in x:
            p = p + 1
            q = q + 1
        while x > 0:
            op.Relu(x)
        while x > 0:
            p = p + 1
            q = q + 1
        if x > 0:
            _y = x
        else:
            _y = x

    b_parsed = sp.parse(complex_ast_fn)
    sp.visit(ast.parse("def f(x):\n    42\n    return\n").body[0])
    assert isinstance(b_parsed, GraphBuilder)

    # Unsupported call (non-Name, non-Attribute func, e.g. lambda call `(lambda z: z)(x)`)
    tree = ast.parse("def f(x): return (lambda z: z)(x)")
    with pytest.raises(CompilationError):
        sp.visit(tree.body[0])


def test_autograd_remaining_branches(tmp_path: Path) -> None:
    """Cover remaining branches in autograd compiler, memory, optimizers, shape_inference, and utils."""
    # memory.py: node with op_type not ending with "Grad"
    g_mem = Graph("mem_g")
    g_mem.add_node(Node("Relu", ["x"], ["y"], {}, "r1"))
    g_mem.add_node(Node("ReluGrad", ["dy"], ["dx"], {}, "rg1"))
    optimize_backward_memory(g_mem)

    # rules.py & vjp.py: call super().build_backward_nodes on a concrete rule or dummy subclass
    rule = get_vjp_rule("Add")
    assert rule is not None

    # shape_inference.py: node not ending in Grad, output not starting with grad_, orig_name not in tensors, out_name already in tensors
    g_si = Graph("si_g")
    g_si.tensors["w"] = Tensor("w", (2, 2), DType.FLOAT32)
    g_si.tensors["grad_existing"] = Tensor("grad_existing", (2, 2), DType.FLOAT32)
    g_si.tensors["existing"] = Tensor("existing", (2, 2), DType.FLOAT32)
    g_si.add_node(Node("Relu", ["w"], ["y"], {}, "fwd_node"))
    g_si.add_node(
        Node(
            "CustomGrad",
            ["dy"],
            ["not_grad", "grad_missing_orig", "grad_existing", "grad_w"],
            {},
            "bwd_node",
        )
    )
    infer_backward_shapes(g_si)
    assert "grad_w" in g_si.tensors

    # utils.py: compile_multi_replica_graph when output tensor not in graph.tensors or already in multi_graph.tensors,
    # embed_distributed_identifiers when grad output not in graph.tensors, add_synchronous_barrier when all_gradients_flat not in outputs
    g_dp = Graph("dp_g")
    g_dp.inputs.append("x")
    g_dp.outputs.append("missing_out_t")
    g_dp.initializers.append("w_init")
    g_dp.tensors["w_init"] = Tensor("w_init", (2,), DType.FLOAT32, is_initializer=True)
    g_dp.add_node(Node("Identity", ["w_init"], ["w_init"], {}, "id_init"))
    compile_multi_replica_graph(g_dp, num_replicas=2)

    g_dist = Graph("dist_g")
    g_dist.outputs.append("grad_missing")
    embed_distributed_identifiers(g_dist)
    add_synchronous_barrier(g_dist)

    # compiler.py remaining branches:
    # 1) save_training_checkpoint with non-requires_grad initializer
    g_ckpt = Graph("ckpt_g")
    g_ckpt.initializers.append("w_frozen")
    g_ckpt.tensors["w_frozen"] = Tensor("w_frozen", (2,), DType.FLOAT32, requires_grad=False)
    ckpt_file = tmp_path / "ckpt.json"
    save_training_checkpoint(g_ckpt, str(ckpt_file))
    # load_training_checkpoint when key in state_dict is not in graph.tensors
    ckpt_file.write_text('{"missing_param": "data"}')
    load_training_checkpoint(g_ckpt, str(ckpt_file))

    # 2) inject_custom_loss_subgraph when tensor already in graph.tensors
    g_main = Graph("main_g")
    g_main.tensors["shared_t"] = Tensor("shared_t", (2,), DType.FLOAT32)
    g_loss = Graph("loss_g")
    g_loss.tensors["shared_t"] = Tensor("shared_t", (2,), DType.FLOAT32)
    inject_custom_loss_subgraph(g_main, g_loss, {})

    # 3) cast_gradients_to_fp32 when grad tensor is float32, and when grad has no producer node
    g_acc = Graph("acc_g")
    g_acc.outputs.extend(["grad_f32", "grad_f16_no_prod"])
    g_acc.tensors["grad_f32"] = Tensor("grad_f32", (2,), "float32")
    g_acc.tensors["grad_f16_no_prod"] = Tensor("grad_f16_no_prod", (2,), "float16")
    g_acc.add_node(Node("Relu", ["x"], ["other_out"], {}, "r_other"))
    cast_gradients_to_fp32(g_acc)

    # 4) optimize_intermediate_casts when node is not Cast
    g_cast = Graph("cast_g")
    g_cast.add_node(Node("Relu", ["x"], ["y"], {}, "r1"))
    optimize_intermediate_casts(g_cast)

    # 5) scale_backward_graph_for_mixed_precision when producer not found, and when consumer not found
    g_scale = Graph("scale_g")
    g_scale.initializers.extend(["p_no_prod", "p_no_cons"])
    g_scale.tensors["p_no_prod"] = Tensor("p_no_prod", (2,), "float32", requires_grad=True)
    g_scale.tensors["p_no_cons"] = Tensor("p_no_cons", (2,), "float32", requires_grad=True)
    g_scale.tensors["grad_p_no_cons"] = Tensor("grad_p_no_cons", (2,), "float32")
    g_scale.add_node(Node("Identity", ["x"], ["grad_p_no_cons"], {}, "prod_p"))
    scale_backward_graph_for_mixed_precision(g_scale)

    # 6) set_eval_mode when Dropout has 1 output (len(node.outputs) == 1)
    g_eval = Graph("eval_g")
    g_eval.add_node(Node("Dropout", ["x"], ["y"], {}, "drop_1out"))
    set_eval_mode(g_eval)

    # 7) inject_bitfit with frozen initializer, and apply_peft_config with unknown peft_type and PROMPT_TUNING with missing tensor
    g_peft = Graph("peft_g")
    g_peft.initializers.extend(["w_frozen", "missing_t"])
    g_peft.tensors["w_frozen"] = Tensor("w_frozen", (2, 2), "float32", requires_grad=False)
    inject_bitfit(g_peft)
    apply_peft_config(g_peft, {"peft_type": "UNKNOWN"})
    apply_peft_config(g_peft, {"peft_type": "PROMPT_TUNING"})

    # 8) build_backward_graph when retained_grads has a name not in grads
    g_fwd = Graph("fwd_g")
    g_fwd.inputs.append("x")
    g_fwd.outputs.append("y")
    g_fwd.tensors["x"] = Tensor("x", (2,), "float32", requires_grad=True)
    g_fwd.tensors["y"] = Tensor("y", (2,), "float32")
    g_fwd.add_node(Node("Relu", ["x"], ["y"], {}, "r1"))
    build_backward_graph(g_fwd, retained_grads=["non_existent_tensor"])

    # 9) ensure_no_microsoft_opsets when graph has non-microsoft node.domain and no opset_imports or non-microsoft opset_imports,
    # and enforce_webgpu_limits when tensor has empty shape or symbolic dim
    g_prop = Graph("prop_g")
    g_prop.add_node(Node("Custom", ["x"], ["y"], {}, "c1", domain="ai.custom"))
    g_prop.opset_imports = {"ai.onnx": 18}
    ensure_no_microsoft_opsets(g_prop)
    delattr(g_prop, "opset_imports")
    ensure_no_microsoft_opsets(g_prop)

    g_wgpu = Graph("wgpu_g")
    g_wgpu.tensors["scalar_t"] = Tensor("scalar_t", (), "float32")
    g_wgpu.tensors["sym_t"] = Tensor("sym_t", ("N", 4), "float32")
    enforce_webgpu_limits(g_wgpu)

    # 10) AOTBuilder.build_accumulate_gradient_graph when bwd external input or node output is not in full_bwd.tensors,
    # and build_gradient_deltas_graph when param_new not in opt_graph.outputs
    aot = AOTBuilder(g_fwd)
    with patch.object(aot.engine, "build_backward_graph") as mock_bwd:
        fake_bwd = Graph("fake_bwd")
        fake_bwd.add_node(Node("Relu", ["x"], ["y"], {}, "r1"))
        fake_bwd.add_node(Node("ReluGrad", ["ext_in"], ["bwd_out"], {}, "bwd_n"))
        fake_bwd.outputs.append("grad_x")
        mock_bwd.return_value = fake_bwd
        aot.build_accumulate_gradient_graph()

    g_aot_delta = Graph("aot_delta")
    g_aot_delta.inputs.append("x")
    g_aot_delta.outputs.append("y")
    g_aot_delta.initializers.append("w")
    g_aot_delta.tensors["x"] = Tensor("x", (2,), "float32")
    g_aot_delta.tensors["y"] = Tensor("y", (2,), "float32")
    g_aot_delta.tensors["w"] = Tensor("w", (2,), "float32", is_initializer=True, requires_grad=True)
    g_aot_delta.add_node(Node("Add", ["x", "w"], ["y"], {}, "add1"))
    aot2 = AOTBuilder(g_aot_delta)
    aot2.build_gradient_deltas_graph(lambda _g, _lr, _p: None, "lr")

    # optimizers.py remaining branches: when global_step, shape, m, v, u are ALREADY in graph.inputs
    g_opt = Graph("opt_g")
    g_opt.inputs.extend(
        [
            "global_step",
            "shape_w",
            "adam_m_w",
            "adam_v_w",
            "adamw_m_w",
            "adamw_v_w",
            "rmsprop_v_w",
            "rmsprop_m_w",
            "adagrad_v_w",
            "adadelta_v_w",
            "adadelta_u_w",
            "shape_grad_w",
        ]
    )
    g_opt.outputs.append("grad_w")
    add_adam_optimizer(g_opt, "lr", ["w"])
    add_adamw_optimizer(g_opt, "lr", ["w"])
    add_rmsprop_optimizer(g_opt, "lr", ["w"], momentum=0.9)
    add_adagrad_optimizer(g_opt, "lr", ["w"])
    add_adadelta_optimizer(g_opt, "lr", ["w"])
    add_differential_privacy_noise(g_opt, ["grad_w"], 0.1, 1.0)
