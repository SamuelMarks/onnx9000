"""Exhaustive tests to reach 100% coverage for onnx9000-optimizer."""

import json
import struct
from pathlib import Path

import numpy as np
from onnx9000.core import onnx_pb2
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import (
    Attribute,
    Constant,
    Graph,
    Node,
    Tensor,
    ValueInfo,
)
from onnx9000.optimizer.hardware.api import optimize
from onnx9000.optimizer.hardware.layout import LayoutOptimizer
from onnx9000.optimizer.hardware.pipeline import PipelineOptimizer
from onnx9000.optimizer.hardware.quantizer import Quantizer
from onnx9000.optimizer.hummingbird.lightgbm_parser import handle_lgbm_objectives
from onnx9000.optimizer.hummingbird.onnxml_parser import parse_onnxml_tree_ensemble
from onnx9000.optimizer.hummingbird.sklearn_parser import (
    parse_random_forest_classifier,
    parse_random_forest_regressor,
)
from onnx9000.optimizer.hummingbird.xgboost_catboost_parser import (
    handle_xgb_objectives,
    parse_xgboost_dump,
)
from onnx9000.optimizer.olive.context import PassContext
from onnx9000.optimizer.olive.model import OliveModel
from onnx9000.optimizer.olive.passes import MixedPrecisionPass
from onnx9000.optimizer.simplifier.api import _calculate_graph_size, simplify
from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass
from onnx9000.optimizer.simplifier.passes.dce import (
    ControlFlowFoldingPass,
    DCEPass,
    IdentityEliminationPass,
)
from onnx9000.optimizer.simplifier.passes.fusion import (
    PatternMatcherFusion,
    fuse_batchnorm_into_conv,
    fuse_batchnorm_into_gemm,
    fuse_consecutive_transpose,
)
from onnx9000.optimizer.simplifier.passes.shapes import (
    ShapeInferencePass,
    extract_rnn_states,
    resolve_dynamic_batch,
    resolve_dynamic_sequence,
)
from onnx9000.optimizer.simplifier.passes.validation import detect_cycles, detect_dangling
from onnx9000.optimizer.simplifier.passes.webgpu import fp16_cast
from onnx9000.optimizer.sparse.calibration import (
    DataLoader,
    cross_entropy_loss,
    evaluate_accuracy,
)
from onnx9000.optimizer.sparse.modifier import (
    AccuracyAwarePruningModifier,
    GlobalMagnitudePruningModifier,
    MagnitudePruningModifier,
    SparseQLinearConvModifier,
)
from onnx9000.optimizer.surgeon.audio import _unpack_scalar, fold_mel_weights
from onnx9000.optimizer.surgeon.fusions import (
    fuse_flash_attention,
    fuse_horizontal_gemm,
)
from onnx9000.optimizer.surgeon.headless import (
    change_batch,
    mutate,
    rename_input,
)
from onnx9000.optimizer.surgeon.layout import optimize_layouts
from onnx9000.optimizer.surgeon.quantization import quantize_ptq


def test_hardware_api_and_layout_branches() -> None:
    """Test remaining branches in hardware/api.py and hardware/layout.py."""
    g = Graph("test_hw")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_tensor(Tensor("x", ("batch", 4), DType.FLOAT32))
    g.add_node(Node("Relu", ["x"], ["y"], {}, "relu1"))
    opt_g = optimize(g, target="cpu")
    assert opt_g is not None

    # inject_transposes_for_layout with target_layout="NCHW" and NHWC Conv
    g_nchw = Graph("g_nchw")
    g_nchw.add_node(Node("Conv", ["x", "w"], ["y"], {"layout": "NHWC"}, "conv1"))
    res_nchw = LayoutOptimizer.inject_transposes_for_layout(g_nchw, target_layout="NCHW")
    assert len(res_nchw.nodes) == 1

    # transpose_cancellation_pass: shared input & Transpose without perm
    g_cancel = Graph("g_cancel")
    g_cancel.add_node(Node("Transpose", ["x"], ["t1"], {"perm": [0, 2, 1]}, "tr1"))
    g_cancel.add_node(Node("Transpose", ["t1"], ["t2"], {}, "tr2"))
    g_cancel.add_node(Node("Relu", ["x"], ["r1"], {}, "r1"))
    LayoutOptimizer.transpose_cancellation_pass(g_cancel)

    # push_transposes_down: shared input, non-elementwise consumer, elementwise without transpose
    g_push = Graph("g_push")
    g_push.add_node(Node("Transpose", ["x"], ["t1"], {"perm": [0, 2, 3, 1]}, "tr1"))
    g_push.add_node(Node("Conv", ["t1", "w"], ["c1"], {}, "c1"))
    g_push.add_node(Node("Add", ["x", "x"], ["a1"], {}, "a1"))
    LayoutOptimizer.push_transposes_down(g_push)

    # update_parameters_for_alignment: MatMul, Reshape without shape, Reshape with -1
    LayoutOptimizer.update_parameters_for_alignment(
        Node("MatMul", ["x", "w"], ["m1"], {}, "m1"), 16
    )
    LayoutOptimizer.update_parameters_for_alignment(Node("Reshape", ["m1"], ["r1"], {}, "r1"), 16)
    LayoutOptimizer.update_parameters_for_alignment(
        Node("Reshape", ["r1"], ["r2"], {"shape": [-1]}, "r2"), 16
    )

    # estimate_vram_usage and chunk_large_tensors_pass with symbolic string dim
    assert LayoutOptimizer.estimate_vram_usage(g) > 0
    chunked = LayoutOptimizer.chunk_large_tensors_pass(g)
    assert chunked is not None


def test_hardware_pipeline_and_quantizer_branches() -> None:
    """Test remaining branches in hardware/pipeline.py and hardware/quantizer.py."""
    g = Graph("pipe")
    g.inputs = ["x"]
    g.add_tensor(Tensor("x", ("batch", 4), DType.FLOAT32))
    g.add_node(Node("Relu", ["unknown_inp"], ["y1"], {}, "n1"))
    g.add_node(Node("Relu", ["x"], ["y2"], {}, "n2"))
    paths = PipelineOptimizer.identify_independent_paths(g)
    assert len(paths) >= 1

    placed = PipelineOptimizer.device_placement_pass(g)
    assert placed is not None
    import random

    random.seed(0)
    for _ in range(5):
        PipelineOptimizer.auto_tuner(g)

    # kl_divergence_calibration where kl >= min_kl on subsequent bins
    data = [np.linspace(0.0, 10.0, 512, dtype=np.float32)]
    low, high = Quantizer.kl_divergence_calibration(data, num_bins=512)
    assert low <= high

    # convert_initializer_to_int8 with non-float initializer (data_type=6)
    tp = onnx_pb2.TensorProto()
    tp.name = "w_int32"
    tp.data_type = 6  # INT32
    tp.raw_data = b"\x00\x00\x00\x00"
    res = Quantizer.convert_initializer_to_int8(tp)
    assert res.data_type == 6


def test_hummingbird_and_olive_branches() -> None:
    """Test remaining branches in hummingbird parsers and olive passes."""
    g = Graph("hb")
    handle_lgbm_objectives(g, "regression")
    handle_xgb_objectives(g, "reg:squarederror")

    tree_node = Node(
        "TreeEnsembleClassifier",
        ["x"],
        ["y"],
        {
            "nodes_treeids": Attribute("nodes_treeids", "INTS", []),
            "nodes_nodeids": Attribute("nodes_nodeids", "INTS", []),
            "nodes_featureids": Attribute("nodes_featureids", "INTS", []),
            "nodes_values": Attribute("nodes_values", "FLOATS", []),
            "nodes_modes": Attribute("nodes_modes", "STRINGS", []),
            "nodes_truenodeids": Attribute("nodes_truenodeids", "INTS", []),
            "nodes_falsenodeids": Attribute("nodes_falsenodeids", "INTS", []),
            "class_treeids": Attribute("class_treeids", "INTS", []),
            "class_nodeids": Attribute("class_nodeids", "INTS", []),
            "class_ids": Attribute("class_ids", "INTS", []),
            "class_weights": Attribute("class_weights", "FLOATS", []),
        },
        "tree",
    )
    assert parse_onnxml_tree_ensemble(tree_node) == []

    assert parse_random_forest_classifier(None) == []
    assert parse_random_forest_regressor(None) == []

    xgb_dumps = [
        json.dumps({"split": "f0", "split_condition": 0.5, "children": []}),
        json.dumps({"split": "f0", "split_condition": 0.5, "children": [{"leaf": 1.0}]}),
    ]
    trees = parse_xgboost_dump(xgb_dumps)
    assert len(trees) == 2

    # MixedPrecisionPass with MatMul
    g_olive = Graph("olive")
    g_olive.add_node(Node("MatMul", ["a", "b"], ["c"], {}, "mm"))
    model = OliveModel(g_olive)
    mp = MixedPrecisionPass()
    mp.run(model, PassContext())
    assert g_olive.nodes[0].attributes["dtype"] == "FLOAT16"


def test_simplifier_api_branches() -> None:
    """Test remaining branches in simplifier/api.py."""
    g = Graph("simp")
    g.initializers = ["missing_init"]
    assert _calculate_graph_size(g) == 0

    g2 = Graph("simp2")
    g2.producer_name = "custom_onnx9000-simplifier"
    g2.inputs = [ValueInfo("x", (1, 4), DType.FLOAT32), "str_inp"]
    g2.outputs = ["y"]
    g2.add_tensor(Tensor("x", (1, 4), DType.FLOAT32))
    g2.add_node(Node("Relu", ["x"], ["y"], {}, "relu"))
    g2.opset_imports = {}
    simplify(
        g2,
        skip_rules=[],
        skip_constant_folding=True,
        skip_fusions=True,
        skip_shape_inference=True,
        input_shapes={"x": ["N", 4], "other": [2, 2]},
        tensor_types={"x": "FLOAT16", "not_in_tensors": "FLOAT32", "other": "UNKNOWN_TYPE"},
        target_opset=18,
    )
    assert g2.opset_imports[""] == 18


def test_constant_folding_branches() -> None:
    """Test remaining branches in simplifier/passes/constant_folding.py."""
    cf = ConstantFoldingPass()

    # Constant node with TENSOR attr where parsed tensor has no data
    g = Graph("cf")
    empty_tp = onnx_pb2.TensorProto()
    empty_tp.name = "empty"
    empty_tp.data_type = onnx_pb2.TensorProto.UNDEFINED
    g.add_node(
        Node(
            "Constant",
            [],
            ["c_out"],
            {"value": Attribute("value", "TENSOR", empty_tp)},
            "c_node",
        )
    )
    cf.run(g)

    # ReduceSum with axes=None and 1 input
    r_node = Node("ReduceSum", ["x"], ["y"], {}, "r_node")
    res = cf._evaluate_node("ReduceSum", [np.array([1.0, 2.0], dtype=np.float32)], {}, r_node)
    assert np.allclose(res, 3.0)

    # SplitToSequence with 1D array of size > 1 (non-scalar split)
    sts_node = Node("SplitToSequence", ["x", "s"], ["y"], {}, "sts")
    assert (
        cf._evaluate_node(
            "SplitToSequence",
            [np.zeros((4, 2)), np.array([2, 2], dtype=np.int64)],
            {},
            sts_node,
        )
        is None
    )

    # ScatterElements with reduction != "none"
    sc_node = Node("ScatterElements", ["d", "i", "u"], ["y"], {}, "sc")
    sc_res = cf._evaluate_node(
        "ScatterElements",
        [
            np.zeros((3,), dtype=np.float32),
            np.array([0], dtype=np.int64),
            np.array([1.0], dtype=np.float32),
        ],
        {"reduction": "add"},
        sc_node,
    )
    assert sc_res is not None

    # NonMaxSuppression with non-overlapping boxes (iou <= iou_threshold) and overlapping boxes (iou > iou_threshold)
    nms_node = Node("NonMaxSuppression", ["boxes", "scores", "max_out", "iou", "score"], ["y"])
    boxes = np.array(
        [[[0.0, 0.0, 1.0, 1.0], [5.0, 5.0, 6.0, 6.0], [0.0, 0.0, 1.0, 1.0]]],
        dtype=np.float32,
    )
    scores = np.array([[[0.9, 0.8, 0.7]]], dtype=np.float32)
    nms_res = cf._evaluate_node(
        "NonMaxSuppression",
        [
            boxes,
            scores,
            np.array([5], dtype=np.int64),
            np.array([0.5], dtype=np.float32),
            np.array([0.1], dtype=np.float32),
        ],
        {},
        nms_node,
    )
    assert len(nms_res) == 2

    # _partial_fold with Sub (non-matching), Where (non-uniform), and Div (non-matching)
    sub_node = Node("Sub", ["x", "y"], ["out"], {}, "s")
    _, changed_sub = cf._partial_fold(sub_node, {})
    assert not changed_sub

    where_node = Node("Where", ["cond", "x", "y"], ["out"], {}, "w")
    _, changed = cf._partial_fold(where_node, {"cond": np.array([True, False])})
    assert not changed

    div_node = Node("Div", ["x", "y"], ["out"], {}, "d")
    _, changed2 = cf._partial_fold(div_node, {})
    assert not changed2


def test_dce_and_identity_elimination_branches() -> None:
    """Test remaining branches in simplifier/passes/dce.py."""
    # DCEPass with unused_inputs_to_prune where all inputs are reachable
    g_dce = Graph("dce")
    g_dce.inputs = ["x"]
    g_dce.outputs = ["y"]
    g_dce.add_node(Node("Relu", ["x"], ["y"], {}, "r"))
    g_dce.value_info = [ValueInfo("dead_vi", (1,), DType.FLOAT32)]
    DCEPass(unused_inputs_to_prune=["x"]).run(g_dce)
    assert len(g_dce.value_info) == 0

    # IdentityEliminationPass branches
    g_id = Graph("id")
    g_id.outputs = [
        "out1",
        "out2",
        "out3",
        "out4",
        "out5",
        "out6",
        "out7",
        "out8",
        "out9",
        "out10",
        "out11",
        "out12",
    ]
    c_raw = Constant("c_raw", values=b"1234", shape=(1,), dtype=DType.FLOAT32)
    c_np = Constant(
        "c_np", values=np.array([1.0], dtype=np.float32), shape=(1,), dtype=DType.FLOAT32
    )
    c_nonzero = Constant(
        "c_nz", values=np.array([2], dtype=np.int64), shape=(1,), dtype=DType.INT64
    )
    g_id.add_tensor(c_raw)
    g_id.add_tensor(c_np)
    g_id.add_tensor(c_nonzero)
    g_id.add_tensor(Tensor("vec", (4,), DType.FLOAT32))

    # 1. Chained Add where inner Add has no initializer
    g_id.add_node(Node("Add", ["x", "y"], ["add_no_init"], {}, "add0"))
    g_id.add_node(Node("Add", ["add_no_init", "c_np"], ["out1"], {}, "add1"))
    # 2. Chained Add where inner Add has raw bytes initializer
    g_id.add_node(Node("Add", ["x", "c_raw"], ["add_raw"], {}, "add2"))
    g_id.add_node(Node("Add", ["add_raw", "c_np"], ["out2"], {}, "add3"))
    # 3. Mul(Add) where inner Add has no initializer
    g_id.add_node(Node("Mul", ["add_no_init", "c_np"], ["out3"], {}, "mul1"))
    # 4. Mul(Add) where inner Add has raw bytes initializer
    g_id.add_node(Node("Mul", ["add_raw", "c_np"], ["out4"], {}, "mul2"))
    # 5. Sqrt(Pow) where Pow has only 1 input
    g_id.add_node(Node("Pow", ["x"], ["pow1"], {}, "pow1"))
    g_id.add_node(Node("Sqrt", ["pow1"], ["out5"], {}, "sqrt1"))
    # 6. Pad with non-zero pads, Pad with non-initializer pads, Tile with non-one repeats, Tile with non-initializer repeats
    g_id.add_node(Node("Pad", ["x", "c_nz"], ["out6"], {}, "pad1"))
    g_id.add_node(Node("Pad", ["x", "dyn_pads"], ["out10"], {}, "pad2"))
    g_id.add_node(Node("Tile", ["x", "c_nz"], ["out7"], {}, "tile1"))
    g_id.add_node(Node("Tile", ["x", "dyn_repeats"], ["out11"], {}, "tile2"))
    # 7. Slice with non-initializer starts, and Slice with non-zero starts
    g_id.add_node(Node("Slice", ["x", "dyn_start", "c_nz"], ["s1"], {}, "slice1"))
    g_id.add_node(Node("Slice", ["s1", "c_nz", "c_nz"], ["out8"], {}, "slice2"))
    # 8. ReduceSum on non-scalar & Unsqueeze(Squeeze) with different axes & Unsqueeze without Squeeze
    g_id.add_node(Node("ReduceSum", ["vec"], ["r_vec"], {}, "r_vec"))
    g_id.add_node(Node("Squeeze", ["r_vec"], ["sq"], {"axes": [0]}, "sq"))
    g_id.add_node(Node("Unsqueeze", ["sq"], ["out9"], {"axes": [1]}, "unsq"))
    g_id.add_node(Node("Unsqueeze", ["r_vec"], ["out12"], {"axes": [0]}, "unsq_no_sq"))
    IdentityEliminationPass().run(g_id)

    # ControlFlowFoldingPass Loop with trip_val >= 10
    g_loop = Graph("loop")
    c_trip = Constant("trip", values=np.array([20], dtype=np.int64), shape=(1,), dtype=DType.INT64)
    g_loop.add_tensor(c_trip)
    g_loop.add_node(Node("Loop", ["trip", ""], ["loop_out"], {}, "loop_node"))
    assert not ControlFlowFoldingPass().run(g_loop)


def test_fusion_shapes_validation_webgpu_branches() -> None:
    """Test remaining branches in fusion.py, shapes.py, validation.py, and webgpu.py."""
    # PatternMatcherFusion non-matching Sigmoid->Mul, Exp->ReduceSum->Div, 3-input Conv->Add
    g_pm = Graph("pm")
    g_pm.add_node(Node("Sigmoid", ["x"], ["sig_out"], {}, "sig"))
    g_pm.add_node(Node("Mul", ["sig_out", "other"], ["mul_out"], {}, "mul"))
    g_pm.add_node(Node("Exp", ["x"], ["exp_out"], {}, "exp"))
    g_pm.add_node(Node("ReduceSum", ["exp_out"], ["rsum_out"], {}, "rsum"))
    g_pm.add_node(Node("Div", ["other", "rsum_out"], ["div_out"], {}, "div"))
    g_pm.add_node(Node("Conv", ["x", "w", "b"], ["conv_out"], {}, "conv"))
    g_pm.add_node(Node("Add", ["conv_out", "b2"], ["add_out"], {}, "add"))
    PatternMatcherFusion().run(g_pm)

    # fuse_batchnorm_into_gemm and fuse_batchnorm_into_conv with downstream consumer node & non-matching graph.outputs
    def make_bn_graph(op_type: str) -> Graph:
        """Test helper function make_bn_graph."""
        g = Graph(f"bn_{op_type}")
        for name, shape in [
            ("w", (2, 2) if op_type == "Gemm" else (2, 2, 1, 1)),
            ("scale", (2,)),
            ("B", (2,)),
            ("mean", (2,)),
            ("var", (2,)),
        ]:
            t = Constant(
                name, values=np.ones(shape, dtype=np.float32), shape=shape, dtype=DType.FLOAT32
            )
            g.add_tensor(t)
            g.initializers.append(name)
        g.outputs = ["other_graph_output"]
        g.add_node(Node(op_type, ["x", "w"], ["op_out"], {}, "op1"))
        g.add_node(
            Node(
                "BatchNormalization", ["op_out", "scale", "B", "mean", "var"], ["bn_out"], {}, "bn1"
            )
        )
        g.add_node(Node("Relu", ["bn_out"], ["relu_out"], {"other_in": "x"}, "relu1"))
        g.nodes[-1].inputs.append("extra_inp")
        return g

    assert fuse_batchnorm_into_gemm(make_bn_graph("Gemm"))
    assert fuse_batchnorm_into_conv(make_bn_graph("Conv"))

    # fuse_consecutive_transpose non-matching cases
    g_tr = Graph("tr")
    g_tr.add_node(Node("Transpose", ["x"], ["t1"], {"perm": [0, 2, 1]}, "tr1"))
    g_tr.add_node(Node("Transpose", ["t1"], ["t2"], {}, "tr2"))
    g_tr.add_node(Node("Transpose", ["t2"], ["t3"], {"perm": [0, 2, 1]}, "tr3"))
    g_tr.add_node(Node("Transpose", ["t3"], ["t4"], {"perm": [2, 0, 1]}, "tr4"))
    g_tr.outputs = ["t4"]
    fuse_consecutive_transpose(g_tr)

    # shapes.py
    g_sh = Graph("sh")
    g_sh.inputs = ["missing_inp", "x3d"]
    g_sh.initializers = ["missing_init"]
    g_sh.add_tensor(Tensor("x3d", (2, 3, 4), DType.FLOAT32))
    g_sh.add_node(Node("Add", ["x3d", "missing_inp"], ["add_out"], {}, "add_missing"))
    g_sh.add_node(Node("MatMul", ["x3d", "x3d"], ["mm_out"], {}, "mm_3d"))
    ShapeInferencePass().run(g_sh)
    resolve_dynamic_batch(g_sh, 4)
    resolve_dynamic_sequence(g_sh, 128)
    extract_rnn_states(g_sh)

    # validation.py: diamond DAG for detect_cycles (visited but not in rec_stack) & empty input in detect_dangling
    g_val = Graph("val")
    g_val.inputs = ["x"]
    g_val.add_node(Node("Relu", ["x"], ["a"], {}, "n1"))
    g_val.add_node(Node("Relu", ["x"], ["b"], {}, "n2"))
    g_val.add_node(Node("Add", ["a", "b", ""], ["c"], {}, "n3"))
    detect_cycles(g_val)
    detect_dangling(g_val)

    # webgpu.py: fp16_cast with non-float32 initializer
    class FakeGraph:
        """Test helper class FakeGraph."""

        def __init__(self) -> None:
            """Test helper function __init__."""
            self.initializers = {"w": Tensor("w", (1,), DType.INT64)}
            self.nodes = []

    assert not fp16_cast(FakeGraph())


def test_sparse_calibration_and_modifier_branches(tmp_path: Path) -> None:
    """Test sparse/calibration.py and remaining branches in sparse/modifier.py."""
    dl = DataLoader([{"x": 1}, {"x": 2}])
    assert list(dl) == [{"x": 1}, {"x": 2}]

    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps([{"x": 3}]))
    dl_file = DataLoader(str(json_path))
    assert list(dl_file) == [{"x": 3}]

    assert cross_entropy_loss([[0.1, 0.9]], [1]) > 0.0
    assert evaluate_accuracy([[0.1, 0.9], [0.9, 0.1]], [1, 1]) == 0.5

    # MagnitudePruningModifier prevent_zeroed_channels where filter is NOT all zeroed
    g_mp = Graph("mp")
    c_mp = Constant(
        "w",
        values=struct.pack("<4f", 10.0, 10.0, 10.0, 10.0),
        shape=(2, 2),
        dtype=DType.FLOAT32,
    )
    g_mp.add_tensor(c_mp)
    mod_mp = MagnitudePruningModifier(
        params=["w"], init_sparsity=0.1, final_sparsity=0.1, prevent_zeroed_channels=True
    )
    mod_mp.apply(g_mp)

    # AccuracyAwarePruningModifier with non-Constant tensor
    g_aa = Graph("aa")
    g_aa.add_tensor(Tensor("t_non_const", (2, 2), DType.FLOAT32))
    mod_aa = AccuracyAwarePruningModifier(params=["t_non_const"])
    assert mod_aa.apply(g_aa, 0.99)

    # GlobalMagnitudePruningModifier with Constant having data=None
    g_gm = Graph("gm")
    g_gm.add_tensor(Constant("w_none", values=None, shape=(2, 2), dtype=DType.FLOAT32))
    mod_gm = GlobalMagnitudePruningModifier(params=["w_none"], final_sparsity=0.5)
    mod_gm.apply(g_gm)

    # SparseQLinearConvModifier with non-Conv node
    g_sq = Graph("sq")
    g_sq.add_node(Node("Relu", ["x"], ["y"], {}, "r"))
    SparseQLinearConvModifier().apply(g_sq)


def test_surgeon_branches(tmp_path: Path) -> None:
    """Test remaining branches in surgeon/*.py."""
    # audio.py: _unpack_scalar with non-bytes/non-numeric data, and fold_mel_weights with non-MelWeightMatrix node
    c_list = Constant("c_list", values=[1.0], shape=(1,), dtype=DType.FLOAT32)
    assert _unpack_scalar(c_list) == 0.0
    g_aud = Graph("aud")
    g_aud.add_node(Node("Relu", ["x"], ["y"], {}, "r"))
    fold_mel_weights(g_aud)

    # fusions.py: fuse_flash_attention non-matching Softmax input and Div input
    g_fa1 = Graph("fa1")
    t_s1 = Tensor("s1", (1, 4), DType.FLOAT32)
    n_soft1 = Node("Softmax", ["raw"], [t_s1], {}, "soft1")
    t_s1.inputs.append(n_soft1)
    g_fa1.add_node(n_soft1)
    g_fa1.add_node(Node("MatMul", [t_s1, "v"], ["out"], {}, "mm1"))
    fuse_flash_attention(g_fa1)

    g_fa2 = Graph("fa2")
    t_d2 = Tensor("d2", (1, 4), DType.FLOAT32)
    n_div2 = Node("Div", ["not_mm", "scale"], [t_d2], {}, "div2")
    t_d2.inputs.append(n_div2)
    t_s2 = Tensor("s2", (1, 4), DType.FLOAT32)
    n_soft2 = Node("Softmax", [t_d2], [t_s2], {}, "soft2")
    t_s2.inputs.append(n_soft2)
    g_fa2.add_node(n_div2)
    g_fa2.add_node(n_soft2)
    g_fa2.add_node(Node("MatMul", [t_s2, "v"], ["out"], {}, "mm2"))
    fuse_flash_attention(g_fa2)

    # fusions.py: HorizontalGemm with non-Gemm node, weight with data=None, and <2 fusible gemms
    g_hg = Graph("hg")
    g_hg.add_tensor(Constant("w1", values=None, shape=(2, 2), dtype=DType.FLOAT32))
    g_hg.add_tensor(Constant("w2", values=b"1234567812345678", shape=(2, 2), dtype=DType.FLOAT32))
    g_hg.add_node(Node("Relu", ["x"], ["r"], {}, "r"))
    g_hg.add_node(Node("Gemm", ["x", "w1"], ["y1"], {}, "g1"))
    g_hg.add_node(Node("Gemm", ["x", "w2"], ["y2"], {}, "g2"))
    g_hg.add_node(Node("Gemm", ["x2", "w1"], ["y3"], {}, "g3"))
    g_hg.add_node(Node("Gemm", ["x2", "not_const"], ["y4"], {}, "g4"))
    fuse_horizontal_gemm(g_hg)

    # headless.py: non-matching inputs/outputs, scalar shapes, and non-remove_node mutation
    g_hl = Graph("hl")
    g_hl.inputs = [ValueInfo("other_in", (), DType.FLOAT32)]
    g_hl.outputs = [ValueInfo("other_out", (), DType.FLOAT32)]
    g_hl.add_node(Node("Relu", ["other_in"], ["other_out"], {}, "r"))
    rename_input(g_hl, "old_in", "new_in")
    change_batch(g_hl, 4)
    script_file = tmp_path / "mut.json"
    script_file.write_text(json.dumps([{"action": "noop"}]))
    mutate(g_hl, str(script_file))

    # layout.py: optimize_layouts with non-NHWC Transpose, missing out_t, and non-pointwise consumer
    g_lay = Graph("lay")
    t_out = Tensor("tr_out", (1, 2, 3, 4), DType.FLOAT32)
    n_conv = Node("Conv", [t_out, "w"], ["c_out"], {}, "conv1")
    g_lay.add_tensor(t_out)
    g_lay.add_node(Node("Transpose", ["x"], ["other"], {}, "tr_no_perm"))
    g_lay.add_node(
        Node(
            "Transpose",
            ["x"],
            ["missing_out_t"],
            {"perm": Attribute("perm", "INTS", [0, 2, 3, 1])},
            "tr_nhwc_missing",
        )
    )
    g_lay.add_node(
        Node(
            "Transpose",
            ["x"],
            ["tr_out"],
            {"perm": Attribute("perm", "INTS", [0, 2, 3, 1])},
            "tr_nhwc",
        )
    )
    g_lay.add_node(n_conv)
    optimize_layouts(g_lay)

    # quantization.py: non-Conv node and non-Constant weight tensor
    g_q = Graph("q")
    g_q.add_tensor(Tensor("w_dyn", (2, 2), DType.FLOAT32))
    g_q.add_node(Node("Relu", ["x"], ["y"], {}, "r"))
    g_q.add_node(Node("MatMul", ["x", "w_dyn"], ["z"], {}, "mm"))
    quantize_ptq(g_q)
