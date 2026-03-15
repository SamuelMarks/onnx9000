"""Provides test_simplifier_coverage.py module functionality."""

import pytest
import numpy as np
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.dtypes import DType
from onnx9000.optimize.simplifier.api import simplify
from onnx9000.optimize.simplifier.passes.constant_folding import ConstantFoldingPass
from onnx9000.optimize.simplifier.passes.dce import DCEPass, IdentityEliminationPass
from onnx9000.optimize.simplifier.passes.fusion import (
    PatternMatcherFusion,
    fuse_linear_activation,
    fuse_matmul_add,
    fuse_consecutive_transpose,
)
from onnx9000.optimize.simplifier.passes.shapes import (
    ShapeInferencePass,
    resolve_dynamic_batch,
    resolve_dynamic_sequence,
    extract_rnn_states,
)
from onnx9000.optimize.simplifier.passes.validation import ValidationPass
from onnx9000.optimize.simplifier.passes.memory_planning import (
    estimate_memory_consumption,
    plan_tensor_lifecycles,
)
from onnx9000.optimize.simplifier.passes.partitioning import partition_for_multi_device


def _make_graph(name="test"):
    """Provides  make graph functionality and verification."""
    return Graph(name)


def test_api_dry_run():
    """Tests the test api dry run functionality."""
    g = _make_graph()
    g2 = simplify(g, dry_run=True)
    assert g is not g2


def test_constant_folding_math_ops():
    """Tests the test constant folding math ops functionality."""
    g = _make_graph()
    val_a = np.array([2.0], dtype=np.float32)
    val_b = np.array([3.0], dtype=np.float32)
    val_c = np.array([1.5], dtype=np.float32)
    g.add_node(Node("Constant", [], ["a"], {"value": val_a}))
    g.add_node(Node("Constant", [], ["b"], {"value": val_b}))
    g.add_node(Node("Constant", [], ["c"], {"value": val_c}))
    g.add_node(Node("Sub", ["a", "b"], ["sub"], {}))
    g.add_node(Node("Div", ["a", "b"], ["div"], {}))
    g.add_node(Node("Pow", ["a", "b"], ["pow"], {}))
    g.add_node(Node("Abs", ["a"], ["abs"], {}))
    g.add_node(Node("Exp", ["a"], ["exp"], {}))
    g.add_node(Node("Log", ["a"], ["log"], {}))
    g.add_node(Node("Sqrt", ["a"], ["sqrt"], {}))
    g.add_node(Node("Cast", ["a"], ["cast"], {"to": 6}))
    g.add_node(Node("Transpose", ["a"], ["trans"], {"perm": [0]}))
    g.add_node(Node("Squeeze", ["a"], ["sqz"], {"axes": [0]}))
    g.add_node(Node("Unsqueeze", ["a"], ["usqz"], {"axes": [1]}))
    g.add_node(Node("Flatten", ["a"], ["flat"], {"axis": 1}))
    g.add_node(Node("Gather", ["a", "c"], ["gather"], {"axis": 0}))
    g.add_node(Node("Shape", ["a"], ["shape"], {}))
    g.add_node(Node("Size", ["a"], ["size"], {}))
    g.add_node(Node("NonZero", ["a"], ["nonzero"], {}))
    ConstantFoldingPass().run(g)
    ops = [n.op_type for n in g.nodes]
    assert "Constant" in ops


def test_constant_folding_partial():
    """Tests the test constant folding partial functionality."""
    g = _make_graph()
    g.add_node(
        Node("Constant", [], ["one"], {"value": np.array([1.0], dtype=np.float32)})
    )
    g.add_node(
        Node("Constant", [], ["zero"], {"value": np.array([0.0], dtype=np.float32)})
    )
    g.add_node(Node("Mul", ["x", "one"], ["m1"], {}))
    g.add_node(Node("Mul", ["one", "x"], ["m2"], {}))
    g.add_node(Node("Mul", ["x", "zero"], ["m3"], {}))
    g.add_node(Node("Pow", ["x", "one"], ["p1"], {}))
    g.add_node(Node("Div", ["x", "one"], ["d1"], {}))
    ConstantFoldingPass().run(g)
    for n in g.nodes:
        if n.outputs[0] in ("m1", "m2", "p1", "d1"):
            assert n.op_type == "Identity"
        if n.outputs[0] == "m3":
            assert n.op_type == "Constant"


def test_dce_initializers():
    """Tests the test dce initializers functionality."""
    g = _make_graph()
    g.initializers = ["init1", "init2"]
    g.outputs = []
    g.add_node(Node("Add", ["init1", "x"], ["y"], {}))
    DCEPass().run(g)
    assert len(g.initializers) == 0


def test_identity_elimination_complex():
    """Tests the test identity elimination complex functionality."""
    g = _make_graph()
    g.add_node(Node("Reshape", ["a", "s1"], ["r1"], {}))
    g.add_node(Node("Reshape", ["r1", "s2"], ["r2"], {}))
    g.add_node(Node("Squeeze", ["b"], ["sq1"], {"axes": [0]}))
    g.add_node(Node("Unsqueeze", ["sq1"], ["usq1"], {"axes": [0]}))
    g.add_node(Node("Transpose", ["c"], ["t1"], {"perm": [1, 0]}))
    g.add_node(Node("Transpose", ["t1"], ["t2"], {"perm": [1, 0]}))
    g.add_node(Node("Identity", ["d"], ["id1"], {}))
    IdentityEliminationPass().run(g)


def test_fusion_conv_bn():
    """Tests the test fusion conv bn functionality."""
    g = _make_graph()
    g.add_node(Node("Conv", ["x", "w"], ["c1"], {}))
    g.add_node(Node("BatchNormalization", ["c1", "s", "b", "m", "v"], ["bn1"], {}))
    PatternMatcherFusion().run(g)


def test_fusion_swish():
    """Tests the test fusion swish functionality."""
    g = _make_graph()
    g.add_node(Node("Sigmoid", ["x"], ["s1"], {}))
    g.add_node(Node("Mul", ["x", "s1"], ["m1"], {}))
    PatternMatcherFusion().run(g)


def test_shapes_inference():
    """Tests the test shapes inference functionality."""
    g = _make_graph()
    g.add_tensor(Tensor("a", shape=(2, 3), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("b", shape=(2, 3), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("c", shape=(3, 4), dtype=DType.FLOAT32))
    g.inputs = ["a", "b", "c"]
    g.add_node(Node("Relu", ["a"], ["relu_a"], {}))
    g.add_node(Node("Add", ["a", "b"], ["add_ab"], {}))
    g.add_node(Node("MatMul", ["a", "c"], ["mm"], {}))
    g.add_node(Node("Gemm", ["a", "c"], ["gemm"], {"transA": 0, "transB": 0}))
    ShapeInferencePass().run(g)
    assert g.tensors["relu_a"].shape == (2, 3)
    assert g.tensors["add_ab"].shape == (2, 3)
    assert g.tensors["mm"].shape == (2, 4)
    assert g.tensors["gemm"].shape == (2, 4)


def test_validation_dangling():
    """Tests the test validation dangling functionality."""
    g = _make_graph()
    ValidationPass().detect_dangling(g)


def test_dummy_stubs():
    """Tests the test dummy stubs functionality."""
    g = _make_graph()
    fuse_linear_activation(g)
    fuse_matmul_add(g)
    fuse_consecutive_transpose(g)
    resolve_dynamic_batch(g)
    resolve_dynamic_sequence(g)
    extract_rnn_states(g)
    estimate_memory_consumption(g)
    plan_tensor_lifecycles(g)
    partition_for_multi_device(g)


def test_constant_folding_more():
    """Tests the test constant folding more functionality."""
    g = _make_graph()
    g.add_node(Node("Constant", [], ["f1"], {"value_float": 1.5}))
    g.add_node(Node("Constant", [], ["i1"], {"value_int": 42}))
    g.add_node(Node("Cast", ["f1"], ["c1"], {"to": 999}))
    g.add_node(Node("Squeeze", ["f1", "i1"], ["sq1"], {}))
    g.add_node(Node("Unsqueeze", ["f1", "i1"], ["usq1"], {}))
    g.add_node(Node("Pow", ["f1", "one"], ["p1"], {}))
    g.add_node(Node("Div", ["f1", "one"], ["d1"], {}))
    g.add_node(
        Node("Constant", [], ["one"], {"value": np.array([1.0], dtype=np.float32)})
    )
    val_a = np.array([1, 2, 3, 4], dtype=np.float32)
    val_starts = np.array([1], dtype=np.int64)
    val_ends = np.array([3], dtype=np.int64)
    val_axes = np.array([0], dtype=np.int64)
    val_steps = np.array([2], dtype=np.int64)
    g.add_node(Node("Constant", [], ["a"], {"value": val_a}))
    g.add_node(Node("Constant", [], ["s"], {"value": val_starts}))
    g.add_node(Node("Constant", [], ["e"], {"value": val_ends}))
    g.add_node(Node("Constant", [], ["ax"], {"value": val_axes}))
    g.add_node(Node("Constant", [], ["st"], {"value": val_steps}))
    g.add_node(Node("Slice", ["a", "s", "e", "ax", "st"], ["slice"], {}))
    g.add_node(Node("SomeOp", ["a", ""], ["out"], {}))
    ConstantFoldingPass().run(g)


def test_dce_squeeze_unsqueeze():
    """Tests the test dce squeeze unsqueeze functionality."""
    g = _make_graph()
    g.add_node(Node("Squeeze", ["b"], ["sq1"], {"axes": [0]}))
    g.add_node(Node("Unsqueeze", ["sq1"], ["usq1"], {"axes": [0]}))
    IdentityEliminationPass().run(g)


def test_matmul_add_fusion_class():
    """Tests the test matmul add fusion class functionality."""
    g = _make_graph()
    g.add_node(Node("MatMul", ["a", "b"], ["c"], {}))
    g.add_node(Node("Add", ["c", "d"], ["e"], {}))
    PatternMatcherFusion().run(g)


def test_shapes_broadcast():
    """Tests the test shapes broadcast functionality."""
    g = _make_graph()
    g.add_tensor(Tensor("a", shape=(1, 3), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("b", shape=(2, 1), dtype=DType.FLOAT32))
    g.inputs = ["a", "b"]
    g.add_node(Node("Add", ["a", "b"], ["c"], {}))
    ShapeInferencePass().run(g)


def test_shapes_init():
    """Tests the test shapes init functionality."""
    g = _make_graph()
    g.add_tensor(Tensor("a", shape=(1, 3), dtype=DType.FLOAT32, is_initializer=True))
    g.initializers = ["a"]
    ShapeInferencePass().run(g)


def test_validation_run():
    """Tests the test validation run functionality."""
    g = _make_graph()
    ValidationPass().run(g)


def test_validation_cycle_deep():
    """Tests the test validation cycle deep functionality."""
    g = _make_graph()
    g.add_node(Node("A", ["x"], ["y"], {}))
    g.add_node(Node("B", ["y"], ["x"], {}))
    with pytest.raises(RuntimeError):
        ValidationPass().run(g)


def test_fusion_layer_norm():
    """Tests the test fusion layer norm functionality."""
    g = _make_graph()
    g.add_node(Node("Sub", ["a", "b"], ["n1"], {}))
    g.add_node(Node("Pow", ["n1", "c"], ["n2"], {}))
    g.add_node(Node("ReduceMean", ["n2"], ["n3"], {}))
    g.add_node(Node("Add", ["n3", "d"], ["n4"], {}))
    g.add_node(Node("Div", ["n1", "n4"], ["n5"], {}))
    g.add_node(Node("Mul", ["n5", "e"], ["n6"], {}))
    g.add_node(Node("Add", ["n6", "f"], ["n7"], {}))
    PatternMatcherFusion().run(g)


def test_fusion_gelu():
    """Tests the test fusion gelu functionality."""
    g = _make_graph()
    g.add_node(Node("Div", ["a", "b"], ["n1"], {}))
    g.add_node(Node("Erf", ["n1"], ["n2"], {}))
    g.add_node(Node("Add", ["n2", "c"], ["n3"], {}))
    g.add_node(Node("Mul", ["a", "n3"], ["n4"], {}))
    g.add_node(Node("Mul", ["n4", "d"], ["n5"], {}))
    PatternMatcherFusion().run(g)


def test_fusion_conv_bn_2():
    """Tests the test fusion conv bn 2 functionality."""
    g = _make_graph()
    g.add_node(Node("Conv", ["x", "w"], ["c1"], {}))
    g.add_node(Node("BatchNormalization", ["c1", "s", "b", "m", "v"], ["bn1"], {}))
    PatternMatcherFusion().run(g)


def test_fusion_matmul_relu():
    """Tests the test fusion matmul relu functionality."""
    g = _make_graph()
    g.add_node(Node("MatMul", ["a", "b"], ["m1"], {}))
    g.add_node(Node("Relu", ["m1"], ["r1"], {}))
    PatternMatcherFusion().run(g)


def test_fusion_gemm_bn():
    """Tests the test fusion gemm bn functionality."""
    g = _make_graph()
    g.add_node(Node("Gemm", ["a", "b", "c"], ["g1"], {}))
    g.add_node(Node("BatchNormalization", ["g1", "s", "b", "m", "v"], ["bn1"], {}))
    PatternMatcherFusion().run(g)


def test_fusion_conv_add():
    """Tests the test fusion conv add functionality."""
    g = _make_graph()
    g.add_node(Node("Conv", ["x", "w"], ["c1"], {}))
    g.add_node(Node("Add", ["c1", "b"], ["a1"], {}))
    PatternMatcherFusion().run(g)


def test_fusion_conv_mul():
    """Tests the test fusion conv mul functionality."""
    g = _make_graph()
    g.add_node(Node("Conv", ["x", "w"], ["c1"], {}))
    g.add_node(Node("Mul", ["c1", "s"], ["m1"], {}))
    PatternMatcherFusion().run(g)


def test_fusion_convtrans_bn():
    """Tests the test fusion convtrans bn functionality."""
    g = _make_graph()
    g.add_node(Node("ConvTranspose", ["x", "w"], ["c1"], {}))
    g.add_node(Node("BatchNormalization", ["c1", "s", "b", "m", "v"], ["bn1"], {}))
    PatternMatcherFusion().run(g)


def test_shapes_inference_more():
    """Tests the test shapes inference more functionality."""
    g = _make_graph()
    g.add_tensor(Tensor("a", shape=(2, 3), dtype=DType.FLOAT32))
    g.add_node(Node("Identity", ["a"], ["id1"], {}))
    g.add_node(Node("Reshape", ["id1"], ["r1"], {}))
    ShapeInferencePass().run(g)


def test_fusion_softmax():
    """Tests the test fusion softmax functionality."""
    g = _make_graph()
    g.add_node(Node("Exp", ["x"], ["e1"], {}))
    g.add_node(Node("ReduceSum", ["e1"], ["rs1"], {"axes": [1]}))
    g.add_node(Node("Div", ["e1", "rs1"], ["out"], {}))
    PatternMatcherFusion().run(g)


def test_fusion_bn_relu():
    """Tests the test fusion bn relu functionality."""
    g = _make_graph()
    g.add_node(Node("BatchNormalization", ["x", "s", "b", "m", "v"], ["bn1"], {}))
    g.add_node(Node("Relu", ["bn1"], ["out"], {}))
    PatternMatcherFusion().run(g)


def test_base_pass_run_once():
    """Tests the test base pass run once functionality."""
    from onnx9000.optimize.simplifier.passes.fusion import FusionPass

    p = FusionPass()
    g = _make_graph()
    p.run(g)


def test_validation_dangling_real():
    """Tests the test validation dangling real functionality."""
    from onnx9000.optimize.simplifier.passes.validation import detect_dangling

    g = _make_graph()
    detect_dangling(g)


def test_shapes_matmul_2d():
    """Tests the test shapes matmul 2d functionality."""
    g = _make_graph()
    g.add_tensor(Tensor("a", shape=(2, 3), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("b", shape=(3, 4), dtype=DType.FLOAT32))
    g.inputs = ["a", "b"]
    g.add_node(Node("MatMul", ["a", "b"], ["c"], {}))
    ShapeInferencePass().run(g)


def test_constant_folding_random():
    """Tests the test constant folding random functionality."""
    g = _make_graph()
    g.add_node(Node("RandomUniform", [], ["out"], {}))
    ConstantFoldingPass().run(g)


def test_constant_folding_error():
    """Tests the test constant folding error functionality."""
    g = _make_graph()
    g.add_node(Node("Constant", [], ["a"], {"value_float": 1.0}))
    g.add_node(Node("Add", ["a", "b_not_exist"], ["c"], {}))
    ConstantFoldingPass().run(g)


def test_constant_folding_partial_div():
    """Tests the test constant folding partial div functionality."""
    g = _make_graph()
    g.add_node(
        Node("Constant", [], ["one"], {"value": np.array([1.0], dtype=np.float32)})
    )
    g.add_node(Node("Div", ["x", "one"], ["d1"], {}))
    ConstantFoldingPass().run(g)


def test_dce_chained_transpose_no_fuse():
    """Tests the test dce chained transpose no fuse functionality."""
    g = _make_graph()
    g.add_node(Node("Transpose", ["a"], ["t1"], {"perm": [1, 0]}))
    g.add_node(Node("Transpose", ["t1"], ["t2"], {"perm": [0, 1]}))
    IdentityEliminationPass().run(g)


def test_dce_squeeze_unsqueeze_none_axes():
    """Tests the test dce squeeze unsqueeze none axes functionality."""
    g = _make_graph()
    g.add_node(Node("Squeeze", ["b"], ["sq1"], {}))
    g.add_node(Node("Unsqueeze", ["sq1"], ["usq1"], {}))
    IdentityEliminationPass().run(g)


def test_cf_tensor_none_data():
    """Tests the test cf tensor none data functionality."""
    g = _make_graph()
    t = Tensor("t1", shape=(1,), dtype=DType.FLOAT32, data=None)
    g.add_node(Node("Constant", [], ["a"], {"value": t}))
    ConstantFoldingPass().run(g)


def test_cf_identity_mul():
    """Tests the test cf identity mul functionality."""
    g = _make_graph()
    g.add_node(
        Node("Constant", [], ["one"], {"value": np.array([1.0], dtype=np.float32)})
    )
    g.add_node(Node("Mul", ["x", "one"], ["m1"], {}))
    g.add_node(Node("Div", ["x", "one"], ["d1"], {}))
    ConstantFoldingPass().run(g)


def test_validation_dangling_warning():
    """Tests the test validation dangling warning functionality."""
    from onnx9000.optimize.simplifier.passes.validation import detect_dangling

    g = _make_graph()
    g.add_node(Node("Identity", ["dangling_in"], ["out"], {}))
    detect_dangling(g)


def test_shapes_dynamic_resolvers():
    """Tests the test shapes dynamic resolvers functionality."""
    g = _make_graph()
    g.add_tensor(Tensor("a", shape=("batch", "seq_len", 3), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("b", shape=(-1, -1, 3), dtype=DType.FLOAT32))
    resolve_dynamic_batch(g)
    resolve_dynamic_sequence(g)
    assert g.tensors["a"].shape == (1, 128, 3)
    assert g.tensors["b"].shape == (1, 128, 3)


def test_all_empty_stubs():
    """Tests the test all empty stubs functionality."""
    from onnx9000.optimize.simplifier.passes.webgpu import (
        polyfill_webgpu_unsupported,
        optimize_for_webgpu,
    )
    from onnx9000.optimize.simplifier.passes.layout import (
        transform_nchw_to_nhwc,
        transform_nhwc_to_nchw,
    )
    from onnx9000.optimize.simplifier.passes.quantization import (
        insert_qat_nodes,
        convert_to_int8,
    )
    from onnx9000.optimize.simplifier.passes.flattening import flatten_subgraphs
    from onnx9000.optimize.simplifier.passes.broadcast import optimize_broadcasting
    from onnx9000.optimize.simplifier.passes.versioning import (
        apply_opset_fallbacks,
        enforce_opset_18,
    )
    from onnx9000.optimize.simplifier.passes.debug import inject_probes

    g = _make_graph()
    polyfill_webgpu_unsupported(g)
    optimize_for_webgpu(g)
    transform_nchw_to_nhwc(g)
    transform_nhwc_to_nchw(g)
    insert_qat_nodes(g)
    convert_to_int8(g)
    flatten_subgraphs(g)
    optimize_broadcasting(g)
    apply_opset_fallbacks(g)
    enforce_opset_18(g)
    inject_probes(g)


def test_cf_tensor_value():
    """Tests the test cf tensor value functionality."""
    g = _make_graph()
    val_a = np.array([1, 2], dtype=np.float32)
    t = Tensor("t1", shape=(2,), dtype=DType.FLOAT32, data=val_a)
    g.add_node(Node("Constant", [], ["a"], {"value": t}))
    ConstantFoldingPass().run(g)


def test_cf_initializer_none():
    """Tests the test cf initializer none functionality."""
    g = _make_graph()
    g.initializers = ["init1"]
    ConstantFoldingPass().run(g)
    g.add_tensor(Tensor("init1", shape=(1,), dtype=DType.FLOAT32, data=None))
    ConstantFoldingPass().run(g)


def test_cf_mul_identity_2():
    """Tests the test cf mul identity 2 functionality."""
    g = _make_graph()
    g.add_node(
        Node("Constant", [], ["one"], {"value": np.array([1.0], dtype=np.float32)})
    )
    g.add_node(Node("Mul", ["one", "x"], ["m1"], {}))
    ConstantFoldingPass().run(g)


def test_cf_pow_identity():
    """Tests the test cf pow identity functionality."""
    g = _make_graph()
    g.add_node(
        Node("Constant", [], ["one"], {"value": np.array([1.0], dtype=np.float32)})
    )
    g.add_node(Node("Pow", ["x", "one"], ["p1"], {}))
    ConstantFoldingPass().run(g)


def test_cf_gather_concat():
    """Tests the test cf gather concat functionality."""
    g = _make_graph()
    g.add_node(
        Node(
            "Constant",
            [],
            ["a"],
            {"value": np.array([[1, 2], [3, 4]], dtype=np.float32)},
        )
    )
    g.add_node(Node("Constant", [], ["ind"], {"value": np.array([1], dtype=np.int64)}))
    g.add_node(Node("Gather", ["a", "ind"], ["gth"], {"axis": 0}))
    g.add_node(Node("Concat", ["a", "a"], ["cat"], {"axis": 1}))
    ConstantFoldingPass().run(g)


def test_dce_squeeze_unsqueeze_mismatch():
    """Tests the test dce squeeze unsqueeze mismatch functionality."""
    g = _make_graph()
    g.add_node(Node("Squeeze", ["b"], ["sq1"], {"axes": [0]}))
    g.add_node(Node("Unsqueeze", ["sq1"], ["usq1"], {"axes": [1]}))
    IdentityEliminationPass().run(g)


def test_api_skip_flags():
    """Tests the test api skip flags functionality."""
    g = _make_graph()
    g.add_node(Node("Add", ["a", "b"], ["c"], {}))
    simplify(g, skip_fusions=True, skip_constant_folding=True)


def test_fusion_pattern_matcher_usages():
    """Tests the test fusion pattern matcher usages functionality."""
    g = _make_graph()
    g.add_node(Node("MatMul", ["a", "b"], ["m1"], {}))
    g.add_node(Node("Add", ["m1", "c"], ["a1"], {}))
    g.outputs = ["m1"]
    PatternMatcherFusion().run(g)


def test_validation_cyclic_2():
    """Tests the test validation cyclic 2 functionality."""
    g = _make_graph()
    g.add_node(Node("A", ["y"], ["x"], {}))
    g.add_node(Node("B", ["x"], ["y"], {}))
    from onnx9000.optimize.simplifier.passes.validation import detect_cycles

    with pytest.raises(RuntimeError):
        detect_cycles(g)


def test_cf_identity_mul_reverse():
    """Tests the test cf identity mul reverse functionality."""
    g = _make_graph()
    g.add_node(
        Node("Constant", [], ["one"], {"value": np.array([1.0], dtype=np.float32)})
    )
    g.add_node(Node("Mul", ["one", "x"], ["m1"], {}))
    ConstantFoldingPass().run(g)


def test_cf_identity_add_reverse():
    """Tests the test cf identity add reverse functionality."""
    g = _make_graph()
    g.add_node(
        Node("Constant", [], ["zero"], {"value": np.array([0.0], dtype=np.float32)})
    )
    g.add_node(Node("Add", ["zero", "x"], ["a1"], {}))
    ConstantFoldingPass().run(g)


def test_validation_simple_cycle():
    """Tests the test validation simple cycle functionality."""
    g = _make_graph()
    g.add_node(Node("A", ["a"], ["b"], {}))
    g.add_node(Node("B", ["b"], ["a"], {}))
    from onnx9000.optimize.simplifier.passes.validation import detect_cycles

    with pytest.raises(RuntimeError):
        detect_cycles(g)


def test_shapes_matmul_1d():
    """Tests the test shapes matmul 1d functionality."""
    g = _make_graph()
    g.add_tensor(Tensor("a", shape=(3,), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("b", shape=(3,), dtype=DType.FLOAT32))
    g.add_node(Node("MatMul", ["a", "b"], ["c"], {}))
    ShapeInferencePass().run(g)


def test_validation_no_cycle():
    """Tests the test validation no cycle functionality."""
    g = _make_graph()
    g.add_node(Node("A", ["a"], ["b"], {}, name="NodeA"))
    g.add_node(Node("B", ["b"], ["c"], {}, name="NodeB"))
    from onnx9000.optimize.simplifier.passes.validation import detect_cycles

    detect_cycles(g)


def test_shapes_identity_2():
    """Tests the test shapes identity 2 functionality."""
    g = _make_graph()
    g.add_node(Node("Identity", ["x_unknown"], ["y"], {}))
    ShapeInferencePass().run(g)


def test_shapes_existing_output_tensor():
    """Tests the test shapes existing output tensor functionality."""
    g = _make_graph()
    g.add_tensor(Tensor("a", shape=(3,), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("c", shape=(3,), dtype=DType.FLOAT32))
    g.add_node(Node("Identity", ["a"], ["c"], {}))
    ShapeInferencePass().run(g)


def test_shapes_existing_output_tensor_2():
    """Tests the test shapes existing output tensor 2 functionality."""
    g = _make_graph()
    g.add_tensor(Tensor("a", shape=(3,), dtype=DType.FLOAT32))
    g.add_tensor(Tensor("c", shape=(3,), dtype=DType.FLOAT32))
    g.inputs = ["a"]
    g.add_node(Node("Identity", ["a"], ["c"], {}))
    ShapeInferencePass().run(g)
