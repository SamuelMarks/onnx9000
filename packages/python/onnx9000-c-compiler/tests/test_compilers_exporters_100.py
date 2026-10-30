"""Exhaustive 100% coverage tests for Python compilers and exporters (Phase 6.3.5)."""

import io
import json
import struct
from pathlib import Path
from unittest.mock import patch

from onnx9000.c_compiler.activations import (
    generate_activation,
    generate_normalization,
    generate_softmax,
)
from onnx9000.c_compiler.ast_builder import C89Builder
from onnx9000.c_compiler.cli import main as c_cli_main
from onnx9000.c_compiler.codegen import BaseCodegenVisitor, CFamilyCodegen, PythonFamilyCodegen
from onnx9000.c_compiler.compiler import C89Compiler
from onnx9000.c_compiler.data_unpacker import unpack_bytes_to_str
from onnx9000.c_compiler.mlir import MLIRCompiler
from onnx9000.c_compiler.operations import (
    generate_einsum,
    generate_elementwise_binary,
    get_strides,
)
from onnx9000.c_compiler.routing import (
    generate_constant_of_shape,
    generate_cumsum,
    generate_pad,
    generate_shape_op,
    generate_slice,
)
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Graph, Node, Tensor, ValueInfo
from onnx9000.onnx2gguf.arch import extract_metadata
from onnx9000.onnx2gguf.compiler import compile_gguf
from onnx9000.onnx2gguf.llama import extract_llama_metadata
from onnx9000.onnx2gguf.reverse import reverse_map_name
from onnx9000.onnx2gguf.tokenizer import extract_tokenizer_metadata
from onnx9000.tflite_exporter.compiler.layout import LayoutOptimizer
from onnx9000.tflite_exporter.compiler.operators import TFLiteOperatorMapping, _map_reshape
from onnx9000.tflite_exporter.compiler.subgraph import compile_graph_to_tflite
from onnx9000.tflite_exporter.exporter import TFLiteExporter
from onnx9000.tflite_exporter.flatbuffer.builder import FlatBufferBuilder
from onnx9000.tflite_exporter.flatbuffer.schema import BuiltinOperator, BuiltinOptions
from onnx9000.tflite_exporter.optimizations.edgetpu import EdgeTPUOptimizer
from onnx9000.tflite_exporter.quantization.quantizer import Quantizer, TensorQuantization


def test_c_compiler_remaining_branches(tmp_path: Path) -> None:
    """Cover remaining lines and branches in onnx9000-c-compiler."""
    b = C89Builder()
    t = Tensor("x", (2, 2), DType.FLOAT32)

    # activations.py: Clip (min+max, min only, max only, neither), PRelu (2 inputs, 1 input), UnknownAct, Softmax axis=0
    generate_activation(b, Node("Softplus", ["x"], ["y"]), t, t, "x", "y", "Softplus")
    generate_activation(b, Node("Clip", ["x", "min_t", "max_t"], ["y"]), t, t, "x", "y", "Clip")
    generate_activation(b, Node("Clip", ["x", "min_t", ""], ["y"]), t, t, "x", "y", "Clip")
    generate_activation(b, Node("Clip", ["x", "", "max_t"], ["y"]), t, t, "x", "y", "Clip")
    generate_activation(b, Node("Clip", ["x"], ["y"]), t, t, "x", "y", "Clip")
    generate_activation(b, Node("PRelu", ["x", "slope"], ["y"]), t, t, "x", "y", "PRelu")
    generate_activation(b, Node("PRelu", ["x"], ["y"]), t, t, "x", "y", "PRelu")
    generate_activation(b, Node("UnknownAct", ["x"], ["y"]), t, t, "x", "y", "UnknownAct")
    generate_softmax(
        b,
        Node("Softmax", ["x"], ["y"], {"axis": Attribute("axis", "INT", 0)}),
        t,
        t,
        "x",
        "y",
    )
    generate_normalization(
        b,
        Node("LayerNormalization", ["x", "s", "b"], ["y"], {"axis": Attribute("axis", "INT", 0)}),
        t,
        t,
        "x",
        "s",
        "b",
        "y",
        "LayerNormalization",
    )

    # codegen.py: BaseCodegenVisitor, CFamilyCodegen, PythonFamilyCodegen
    g_cg = Graph("cg")
    g_cg.add_node(Node("Relu", ["x"], ["y"], {}, "r1"))
    base_v = BaseCodegenVisitor()
    assert base_v.get_var_name("tmp") == "tmp1"
    assert "Unknown operation: Relu" in base_v.visit(g_cg)

    c_v = CFamilyCodegen()
    assert "void forward_cg()" in c_v.visit(g_cg)

    py_v = PythonFamilyCodegen()
    py_v.imports.add("math")
    assert "import math" in py_v.visit(g_cg)

    # compiler.py: DType.STRING initializer and missing tensor in arena.tensor_offsets
    g_str = Graph("str_g")
    t_str_list = Tensor("s_list", (2,), DType.STRING, is_initializer=True)
    t_str_list.data = [b"hello\n", 'wor"ld']
    t_str_single = Tensor("s_single", (1,), DType.STRING, is_initializer=True, data=b"one")
    g_str.add_tensor(t_str_list)
    g_str.add_tensor(t_str_single)
    g_str.initializers.extend(["s_list", "s_single"])
    comp = C89Compiler(g_str)
    comp.arena.tensor_offsets["non_existent_tensor"] = 0
    _, c_src = comp.generate()
    assert "weights_s_list" in c_src and "weights_s_single" in c_src

    # cli.py: --no-opt and --quiet with non-arduino target, and --target arduino --quiet without --no-opt
    model_file = tmp_path / "m.onnx"
    model_file.write_bytes(b"dummy")
    out_dir = tmp_path / "c_out"
    with (
        patch("onnx9000.c_compiler.cli.load", return_value=Graph("cli_g")),
        patch("sys.argv", ["onnx2c", str(model_file), "-o", str(out_dir), "--no-opt", "--quiet"]),
    ):
        c_cli_main()
        assert (out_dir / "CMakeLists.txt").exists()
    with (
        patch("onnx9000.c_compiler.cli.load", return_value=Graph("cli_g2")),
        patch(
            "sys.argv",
            ["onnx2c", str(model_file), "-o", str(out_dir), "--target", "arduino", "--quiet"],
        ),
    ):
        c_cli_main()

    # data_unpacker.py: BOOL, BFLOAT16, FLOAT64 (with force_float32=False)
    assert "0x05" in unpack_bytes_to_str(b"\x01\x00\x01", DType.BOOL)
    assert "1.0f" in unpack_bytes_to_str(struct.pack("<H", 0x3F80), DType.BFLOAT16)
    f64_str = unpack_bytes_to_str(struct.pack("<dd", 1.5, 2.0), DType.FLOAT64, force_float32=False)
    assert "1.5" in f64_str and "2.0" in f64_str

    # mlir.py: node that is neither Conv nor Relu
    g_mlir = Graph("mlir_g")
    g_mlir.add_node(Node("Add", ["a", "b"], ["c"], {}, "add1"))
    assert "module {" in MLIRCompiler(g_mlir).generate_tosa()

    # operations.py: get_strides([]), non-pow2 integer division, Einsum bytes/Attribute/non-str and mk,kn->mn
    assert get_strides([]) == []
    t_div = Tensor("d3", (1,), DType.INT32, is_initializer=True, data=struct.pack("<i", 3))
    generate_elementwise_binary(
        b, Node("Div", ["x", "d3"], ["y"]), "/", t, t, t_div, "x", "d3", "y"
    )
    generate_einsum(
        b,
        Node("Einsum", ["a", "b"], ["c"], {"equation": b"mk,kn->mn"}),
        t,
        [t, t],
        "c",
        ["a", "b"],
    )
    generate_einsum(
        b,
        Node(
            "Einsum",
            ["a", "b"],
            ["c"],
            {"equation": Attribute("equation", "STRING", b"mk,kn->mn")},
        ),
        t,
        [t, t],
        "c",
        ["a", "b"],
    )
    generate_einsum(
        b,
        Node("Einsum", ["a", "b"], ["c"], {"equation": 12345}),
        t,
        [t, t],
        "c",
        ["a", "b"],
    )

    # routing.py: generate_shape_op (out_name == in_name), generate_pad (3D), generate_slice (1, 2, 3 and 4 inputs),
    # generate_constant_of_shape (empty data and BOOL data), generate_cumsum (1 input)
    generate_shape_op(b, Node("Reshape", ["x"], ["x"]), t, t, "x", "x", "Reshape")
    t3d = Tensor("x3", (2, 2, 2), DType.FLOAT32)
    generate_pad(b, Node("Pad", ["x3"], ["y3"]), t3d, t3d, None, None, "x3", "", "", "y3")
    t1d = Tensor("s1", (1,), DType.INT32)
    generate_slice(b, Node("Slice", ["x"], ["y"]), t, [t], ["x"], "y")
    generate_slice(b, Node("Slice", ["x", "s"], ["y"]), t, [t, t1d], ["x", "s"], "y")
    generate_slice(b, Node("Slice", ["x", "s", "e"], ["y"]), t, [t, t1d, t1d], ["x", "s", "e"], "y")
    generate_slice(
        b,
        Node("Slice", ["x", "s", "e", "a"], ["y"]),
        t,
        [t, t1d, t1d, t1d],
        ["x", "s", "e", "a"],
        "y",
    )
    generate_constant_of_shape(
        b,
        Node(
            "ConstantOfShape",
            ["s1"],
            ["y"],
            {
                "value": Attribute(
                    "value", "TENSOR", Tensor("empty_v", (1,), DType.FLOAT32, data=b"")
                )
            },
        ),
        t,
        [t1d],
        ["s1"],
        "y",
    )
    generate_constant_of_shape(
        b,
        Node(
            "ConstantOfShape",
            ["s1"],
            ["y"],
            {
                "value": Attribute(
                    "value", "TENSOR", Tensor("bool_v", (1,), DType.BOOL, data=b"\x01")
                )
            },
        ),
        t,
        [t1d],
        ["s1"],
        "y",
    )
    generate_cumsum(b, Node("CumSum", ["x"], ["y"]), t, [t], ["x"], "y")


def test_onnx2gguf_remaining_branches() -> None:
    """Cover remaining branches in onnx9000-onnx2gguf."""
    g = Graph("llama_g")
    g.add_node(Node("Relu", ["x"], ["y"], {}, "r1"))
    g.tensors["unmatched_weight"] = Tensor("unmatched_weight", (4, 4), DType.FLOAT32)
    assert extract_metadata(g, arch_override="llama")["llama.context_length"] == 2048
    assert extract_metadata(g, arch_override="phi2")["phi2.context_length"] == 2048
    assert extract_llama_metadata(g)["custom.is_swiglu"] is False
    assert reverse_map_name("blk.0.unknown_suffix.weight") == "blk.0.unknown_suffix.weight"
    assert "tokenizer.ggml.tokens" not in extract_tokenizer_metadata(
        json.dumps({"model": {"vocab": []}})
    )

    g.initializers.extend(["missing_init", "non_tensor_init", "no_data_init"])
    g.tensors["non_tensor_init"] = "not_a_tensor"
    g.tensors["no_data_init"] = Tensor("no_data_init", (2,), DType.FLOAT32, data=None)
    out = io.BytesIO()
    compile_gguf(
        g,
        out,
        kv_overrides={
            "tokenizer.ggml.model": "custom_tok",
            "tokenizer.json": json.dumps({"model": {"vocab": {}}}),
        },
    )
    assert len(out.getvalue()) > 0


def test_tflite_exporter_remaining_branches() -> None:
    """Cover remaining branches in onnx9000-tflite-exporter."""
    # flatbuffer/builder.py default value skips, slot when vtable is None, and finish without identifier
    fb = FlatBufferBuilder()
    fb.slot(0)
    fb.start_object(4)
    fb.add_field_int16(0, 0, 0)
    fb.add_field_int64(1, 0, 0)
    fb.add_field_float32(2, 0.0, 0.0)
    fb.add_field_float64(3, 0.0, 0.0)
    root = fb.end_object()
    fb.finish(root, identifier=None)
    assert len(fb.as_bytearray()) > 0

    # exporter.py: explicit version != -1
    exp = TFLiteExporter()
    assert exp.get_or_add_operator_code(BuiltinOperator.ADD, version=2) == 0

    # compiler/operators.py: _map_reshape with non-initializer shape tensor
    g_op = Graph("op_g")
    g_op.tensors["dyn_shape"] = Tensor("dyn_shape", (2,), "int64", is_initializer=False)
    _map_reshape(fb, Node("Reshape", ["x", "dyn_shape"], ["y"]), g_op)

    # compiler/subgraph.py: CUSTOM operator with non-bytes custom_options
    g_sub = Graph("sub_g")
    g_sub.inputs.append(ValueInfo("x", (1, 2), "float32"))
    g_sub.outputs.append(ValueInfo("y", (1, 2), "float32"))
    g_sub.tensors["x"] = Tensor("x", (1, 2), "float32")
    g_sub.tensors["y"] = Tensor("y", (1, 2), "float32")
    g_sub.add_node(
        Node(
            "CustomOp",
            ["x"],
            ["y"],
            {"custom_options": Attribute("custom_options", "STRING", "not_bytes")},
            "c1",
        )
    )
    with patch(
        "onnx9000.tflite_exporter.compiler.operators.map_onnx_node_to_tflite",
        return_value=TFLiteOperatorMapping(BuiltinOperator.CUSTOM, BuiltinOptions.NONE, None),
    ):
        compile_graph_to_tflite(g_sub, TFLiteExporter(), keep_nchw=True)

    # optimizations/edgetpu.py branches
    g_tpu = Graph("tpu_g")
    g_tpu.value_info.append(ValueInfo("vi_int8", (1, 4, 4), "int8"))
    g_tpu.tensors["w_2d"] = Tensor("w_2d", (4, 4), "float32")
    g_tpu.tensors["w_4d_aligned"] = Tensor("w_4d_aligned", (4, 4, 3, 3), "float32")
    g_tpu.tensors["static_starts"] = Tensor("static_starts", (1,), "int64")
    g_tpu.add_node(Node("Conv", ["vi_int8", "missing_w"], ["y1"], {}, "c_missing"))
    g_tpu.add_node(Node("Conv", ["vi_int8", "w_2d"], ["y2"], {}, "c_2d"))
    g_tpu.add_node(Node("Conv", ["vi_int8", "w_4d_aligned"], ["y3"], {}, "c_aligned"))
    g_tpu.add_node(Node("Conv", [], ["y4"], {}, "c_no_in"))
    g_tpu.add_node(Node("Slice", ["vi_int8", "static_starts"], ["y5"], {}, "slice_static"))
    EdgeTPUOptimizer(g_tpu).optimize()

    # quantization/quantizer.py branches
    q_none = Quantizer(Graph("q_g"), mode="other")
    q_none.quantize()
    q_none.quantization_map["q_t"] = TensorQuantization(
        min=[0.0], max=[], scale=[], zero_point=[], quantized_dimension=0
    )
    q_none.get_quantization_offset(fb, Tensor("q_t", (1,), "float32"))

    g_q8 = Graph("q8_g")
    g_q8.add_node(Node("QuantizeLinear", [], ["y"], {}, "ql_empty"))
    g_q8.add_node(Node("QuantizeLinear", ["x", "s_missing", "z_missing"], ["y"], {}, "ql_missing"))
    Quantizer(g_q8, mode="int8").quantize()

    # compiler/layout.py remaining branches
    g_lay = Graph("lay_g")
    g_lay.inputs.append(ValueInfo("in_3d", (1, 3, 8), "float32"))
    g_lay.inputs.append(ValueInfo("in_noshape", (), "float32"))
    g_lay.tensors["w_4d"] = Tensor("w_4d", (3, 3, 1, 3), "float32")
    # Conv on 3D input with no outputs, and Conv on 3D input with 4D weight
    g_lay.add_node(Node("Conv", ["in_3d"], [], {}, "conv_no_out"))
    g_lay.add_node(Node("Conv", ["in_3d", "w_4d"], ["out_3d"], {}, "conv_w4d"))
    opt = LayoutOptimizer(g_lay, keep_nchw=False)
    opt.expand_1d_spatial_ops()

    # fuse_conv_batch_normalization when Conv has missing w_tensor, 1 input, or w_tensor channels != scale_data len, or consumer_idx < i
    g_bn = Graph("bn_g")
    g_bn.tensors["s"] = Tensor("s", (2,), "float32", data=struct.pack("<2f", 1.0, 1.0))
    g_bn.tensors["b"] = Tensor("b", (2,), "float32", data=struct.pack("<2f", 0.0, 0.0))
    g_bn.tensors["m"] = Tensor("m", (2,), "float32", data=struct.pack("<2f", 0.0, 0.0))
    g_bn.tensors["v"] = Tensor("v", (2,), "float32", data=struct.pack("<2f", 1.0, 1.0))
    g_bn.tensors["w_mismatch"] = Tensor(
        "w_mismatch", (1, 1, 1, 1), "float32", data=struct.pack("<f", 1.0)
    )
    g_bn.tensors["w_match"] = Tensor(
        "w_match", (2, 1, 1, 1), "float32", data=struct.pack("<2f", 1.0, 1.0)
    )
    g_bn.tensors["b_empty"] = Tensor("b_empty", (2,), "float32", data=b"")
    # BN before Conv so consumer_idx <= i
    g_bn.add_node(Node("BatchNormalization", ["c_out", "s", "b", "m", "v"], ["bn_out"], {}, "bn0"))
    g_bn.add_node(Node("Conv", ["x", "w_match", "b_empty"], ["c_out"], {}, "conv0"))
    g_bn.add_node(Node("Conv", ["x"], ["c_1in"], {}, "conv_1in"))
    g_bn.add_node(Node("BatchNormalization", ["c_1in", "s", "b", "m", "v"], ["bn1"], {}, "bn1"))
    g_bn.add_node(Node("Conv", ["x", "w_missing_bn"], ["c_wmiss"], {}, "conv_wmiss"))
    g_bn.add_node(
        Node("BatchNormalization", ["c_wmiss", "s", "b", "m", "v"], ["bn_wmiss"], {}, "bn_wmiss")
    )
    g_bn.add_node(Node("Conv", ["x", "w_mismatch"], ["c_mis"], {}, "conv_mis"))
    g_bn.add_node(Node("BatchNormalization", ["c_mis", "s", "b", "m", "v"], ["bn2"], {}, "bn2"))
    # BN with another consumer of c_other on input 1
    g_bn.add_node(Node("Conv", ["x", "w_match"], ["c_other"], {}, "conv_other"))
    g_bn.add_node(
        Node("BatchNormalization", ["other_x", "c_other", "b", "m", "v"], ["bn3"], {}, "bn3")
    )
    LayoutOptimizer(g_bn, keep_nchw=False).fuse_conv_batch_normalization()

    # decompose_batch_normalization with < 5 inputs, no outputs, or missing tensor data
    g_dec = Graph("dec_g")
    g_dec.add_node(Node("BatchNormalization", ["x"], ["y"], {}, "bn_short"))
    g_dec.add_node(Node("BatchNormalization", ["x", "s", "b", "m", "v"], [], {}, "bn_no_out"))
    g_dec.add_node(Node("BatchNormalization", ["x", "s", "b", "m", "v"], ["y2"], {}, "bn_no_t"))
    LayoutOptimizer(g_dec, keep_nchw=False).decompose_batch_normalization()

    # evaluate_constants with Div having 1 input, non-initializer divisor, or non-zero divisor
    g_div = Graph("div_g")
    g_div.tensors["b_non_init"] = Tensor("b_non_init", (1,), "float32", is_initializer=False)
    g_div.tensors["b_nonzero"] = Tensor(
        "b_nonzero", (1,), "float32", is_initializer=True, data=struct.pack("<f", 2.0)
    )
    g_div.add_node(Node("Div", ["x"], ["y1"], {}, "div_1in"))
    g_div.add_node(Node("Div", ["x", "b_non_init"], ["y2"], {}, "div_non_init"))
    g_div.add_node(Node("Div", ["x", "b_nonzero"], ["y3"], {}, "div_nonzero"))
    LayoutOptimizer(g_div, keep_nchw=False).evaluate_constants()

    # inject_transposes with empty inputs/outputs and tensor shape empty
    g_inj = Graph("inj_g")
    g_inj.tensors["t_empty_shape"] = Tensor("t_empty_shape", (), "float32")
    g_inj.add_node(Node("Conv", [], [], {}, "conv_empty_io"))
    g_inj.add_node(Node("Conv", ["t_empty_shape"], ["y"], {}, "conv_empty_shape"))
    LayoutOptimizer(g_inj, keep_nchw=False).inject_transposes()

    # push_down_transposes when producer_idx > i, or axis out of [0, 4)
    g_push = Graph("push_g")
    g_push.add_node(
        Node(
            "Softmax",
            ["t_out"],
            ["s_out"],
            {"axis": Attribute("axis", "INT", 5)},
            "soft_before_trans",
        )
    )
    g_push.add_node(
        Node(
            "Transpose",
            ["x"],
            ["t_out"],
            {"perm": Attribute("perm", "INTS", [0, 2, 3, 1])},
            "trans_after_soft",
        )
    )
    LayoutOptimizer(g_push, keep_nchw=False).push_down_transposes()

    # cancel_transposes when perm is missing, outputs[0] != inputs[0], or perms don't cancel
    g_can = Graph("can_g")
    g_can.add_node(Node("Transpose", ["x0"], ["t0"], {}, "tr_no_perm"))
    g_can.add_node(
        Node("Transpose", ["t0"], ["t1"], {"perm": Attribute("perm", "INTS", [0, 2, 3, 1])}, "tr1")
    )
    g_can.add_node(
        Node(
            "Transpose",
            ["other"],
            ["t2"],
            {"perm": Attribute("perm", "INTS", [0, 3, 1, 2])},
            "tr2",
        )
    )
    g_can.add_node(
        Node("Transpose", ["t2"], ["t3"], {"perm": Attribute("perm", "INTS", [0, 3, 1, 2])}, "tr3")
    )
    LayoutOptimizer(g_can, keep_nchw=False).cancel_transposes()

    # fuse_activations_and_matmuls when MatMul consumer is not Add, Add has only [y, y], or Add is before MatMul, or Relu is before Conv
    g_fuse = Graph("fuse_g")
    g_fuse.add_node(Node("MatMul", ["x", "w"], ["mm_relu_in"], {}, "mm_to_relu"))
    g_fuse.add_node(Node("Relu", ["mm_relu_in"], ["mm_relu_out"], {}, "relu_of_mm"))
    g_fuse.add_node(Node("Add", ["mm_out", "b"], ["add_out"], {}, "add_before"))
    g_fuse.add_node(Node("MatMul", ["x", "w"], ["mm_out"], {}, "mm_after"))
    g_fuse.add_node(Node("MatMul", ["x", "w"], ["mm_self"], {}, "mm_self_node"))
    g_fuse.add_node(Node("Add", ["mm_self", "mm_self"], ["add_self"], {}, "add_self_node"))
    g_fuse.add_node(Node("Relu", ["conv_out"], ["relu_out"], {}, "relu_before"))
    g_fuse.add_node(Node("Conv", ["x", "w"], ["conv_out"], {}, "conv_after"))
    LayoutOptimizer(g_fuse, keep_nchw=False).fuse_activations_and_matmuls()

    # fold_constants: Gemm with transB=1, Resize with non-int64 sizes, Resize with missing scales, Resize with in_name in value_info, Conv with missing weight_tensor
    g_fold = Graph("fold_g")
    g_fold.inputs.append(ValueInfo("x_in", (1, 3, 4, 4), "float32"))
    g_fold.value_info.append(ValueInfo("x_vi", (1, 3, 4, 4), "float32"))
    g_fold.tensors["sizes_i32"] = Tensor("sizes_i32", (4,), "int32")
    g_fold.tensors["scales_dyn"] = Tensor("scales_dyn", (4,), "float32", is_initializer=False)
    g_fold.tensors["scales_init"] = Tensor(
        "scales_init",
        (4,),
        "float32",
        is_initializer=True,
        data=struct.pack("<4f", 1.0, 1.0, 2.0, 2.0),
    )
    g_fold.add_node(
        Node(
            "Gemm",
            ["x_in", "missing_gemm_w"],
            ["g1"],
            {"transB": Attribute("transB", "INT", 1)},
            "gemm_tb1",
        )
    )
    g_fold.add_node(Node("Resize", ["x_in", "", "", "sizes_i32"], ["r1"], {}, "res_i32"))
    g_fold.add_node(Node("Resize", ["x_vi", "", "scales_dyn"], ["r2"], {}, "res_dyn_scales"))
    g_fold.add_node(Node("Resize", ["x_in", "", "scales_init"], ["r3"], {}, "res_short_inputs"))
    g_fold.add_node(
        Node("Resize", ["x_in", "", "missing_scales"], ["r4"], {}, "res_missing_scales")
    )
    g_fold.add_node(Node("Conv", ["x_in", "missing_conv_w"], ["c1"], {}, "conv_missing_w"))
    g_fold.add_node(
        Node(
            "ConvTranspose",
            ["x_in"],
            ["ct1"],
            {"output_padding": Attribute("output_padding", "INTS", [])},
            "ct_1in",
        )
    )
    LayoutOptimizer(g_fold, keep_nchw=False).fold_constants()

    # rewrite_negative_axes with empty inputs or empty shape; recalculate_shapes without perm
    g_neg = Graph("neg_g")
    g_neg.tensors["t_empty"] = Tensor("t_empty", (), "float32")
    g_neg.add_node(
        Node(
            "Softmax",
            [],
            ["y1"],
            {
                "axis": Attribute("axis", "INT", -1),
                "axes": Attribute("axes", "INTS", [-1]),
            },
            "s_no_in",
        )
    )
    g_neg.add_node(
        Node(
            "Softmax",
            ["t_empty"],
            ["y2"],
            {
                "axis": Attribute("axis", "INT", -1),
                "axes": Attribute("axes", "INTS", [-1]),
            },
            "s_empty_shape",
        )
    )
    g_neg.add_node(Node("Transpose", ["t_empty"], ["y3"], {}, "tr_no_perm"))
    opt_neg = LayoutOptimizer(g_neg, keep_nchw=False)
    opt_neg.rewrite_negative_axes()
    opt_neg.recalculate_shapes()
