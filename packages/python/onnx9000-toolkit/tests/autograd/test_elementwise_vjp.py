"""Module providing core logic and structural definitions."""

from onnx9000.core.ir import Node
from onnx9000.toolkit.training.autograd.rules import get_vjp_rule


def test_sub_vjp() -> None:
    """Tests the test_sub_vjp functionality."""
    node = Node("Sub", ["a", "b"], ["c"], {}, name="sub_node")
    rule = get_vjp_rule("Sub")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2
    assert nodes[0].op_type == "Identity"
    assert nodes[1].op_type == "Neg"
    assert names == ["grad_a_wrt_sub_node", "grad_b_wrt_sub_node"]


def test_div_vjp() -> None:
    """Tests the test_div_vjp functionality."""
    node = Node("Div", ["a", "b"], ["c"], {}, name="div_node")
    rule = get_vjp_rule("Div")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 5
    assert names == ["grad_a_wrt_div_node", "grad_b_wrt_div_node"]


def test_pow_vjp() -> None:
    """Tests the test_pow_vjp functionality."""
    node = Node("Pow", ["a", "b"], ["c"], {}, name="pow_node")
    rule = get_vjp_rule("Pow")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2


def test_mod_vjp() -> None:
    """Tests the test_mod_vjp functionality."""
    node = Node("Mod", ["a", "b"], ["c"], {}, name="mod_node")
    rule = get_vjp_rule("Mod")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2


def test_abs_vjp() -> None:
    """Tests the test_abs_vjp functionality."""
    node = Node("Abs", ["a"], ["c"], {}, name="abs_node")
    rule = get_vjp_rule("Abs")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2


def test_neg_vjp() -> None:
    """Tests the test_neg_vjp functionality."""
    node = Node("Neg", ["a"], ["c"], {}, name="neg_node")
    rule = get_vjp_rule("Neg")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_sign_vjp() -> None:
    """Tests the test_sign_vjp functionality."""
    node = Node("Sign", ["a"], ["c"], {}, name="sign_node")
    rule = get_vjp_rule("Sign")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_exp_vjp() -> None:
    """Tests the test_exp_vjp functionality."""
    node = Node("Exp", ["a"], ["c"], {}, name="exp_node")
    rule = get_vjp_rule("Exp")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_log_vjp() -> None:
    """Tests the test_log_vjp functionality."""
    node = Node("Log", ["a"], ["c"], {}, name="log_node")
    rule = get_vjp_rule("Log")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_sqrt_vjp() -> None:
    """Tests the test_sqrt_vjp functionality."""
    node = Node("Sqrt", ["a"], ["c"], {}, name="sqrt_node")
    rule = get_vjp_rule("Sqrt")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_sin_vjp() -> None:
    """Tests the test_sin_vjp functionality."""
    node = Node("Sin", ["a"], ["c"], {}, name="sin_node")
    rule = get_vjp_rule("Sin")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2


def test_cos_vjp() -> None:
    """Tests the test_cos_vjp functionality."""
    node = Node("Cos", ["a"], ["c"], {}, name="cos_node")
    rule = get_vjp_rule("Cos")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 3


def test_tan_vjp() -> None:
    """Tests the test_tan_vjp functionality."""
    node = Node("Tan", ["a"], ["c"], {}, name="tan_node")
    rule = get_vjp_rule("Tan")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_asin_vjp() -> None:
    """Tests the test_asin_vjp functionality."""
    node = Node("Asin", ["a"], ["c"], {}, name="asin_node")
    rule = get_vjp_rule("Asin")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_acos_vjp() -> None:
    """Tests the test_acos_vjp functionality."""
    node = Node("Acos", ["a"], ["c"], {}, name="acos_node")
    rule = get_vjp_rule("Acos")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_atan_vjp() -> None:
    """Tests the test_atan_vjp functionality."""
    node = Node("Atan", ["a"], ["c"], {}, name="atan_node")
    rule = get_vjp_rule("Atan")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_sinh_vjp() -> None:
    """Tests the test_sinh_vjp functionality."""
    node = Node("Sinh", ["a"], ["c"], {}, name="sinh_node")
    rule = get_vjp_rule("Sinh")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2


def test_cosh_vjp() -> None:
    """Tests the test_cosh_vjp functionality."""
    node = Node("Cosh", ["a"], ["c"], {}, name="cosh_node")
    rule = get_vjp_rule("Cosh")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2


def test_asinh_vjp() -> None:
    """Tests the test_asinh_vjp functionality."""
    node = Node("Asinh", ["a"], ["c"], {}, name="asinh_node")
    rule = get_vjp_rule("Asinh")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_acosh_vjp() -> None:
    """Tests the test_acosh_vjp functionality."""
    node = Node("Acosh", ["a"], ["c"], {}, name="acosh_node")
    rule = get_vjp_rule("Acosh")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_atanh_vjp() -> None:
    """Tests the test_atanh_vjp functionality."""
    node = Node("Atanh", ["a"], ["c"], {}, name="atanh_node")
    rule = get_vjp_rule("Atanh")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_erf_vjp() -> None:
    """Tests the test_erf_vjp functionality."""
    node = Node("Erf", ["a"], ["c"], {}, name="erf_node")
    rule = get_vjp_rule("Erf")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_isnan_vjp() -> None:
    """Tests the test_isnan_vjp functionality."""
    node = Node("IsNaN", ["a"], ["c"], {}, name="isnan_node")
    rule = get_vjp_rule("IsNaN")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1


def test_reciprocal_vjp() -> None:
    node = Node("Reciprocal", ["a"], ["c"], {}, name="reciprocal_node")
    rule = get_vjp_rule("Reciprocal")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "ReciprocalGrad"


def test_clip_vjp() -> None:
    node = Node("Clip", ["a", "min", "max"], ["c"], {}, name="clip_node")
    rule = get_vjp_rule("Clip")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "ClipGrad"


def test_round_vjp() -> None:
    node = Node("Round", ["a"], ["c"], {}, name="round_node")
    rule = get_vjp_rule("Round")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "ConstantOfShape"


def test_floor_vjp() -> None:
    node = Node("Floor", ["a"], ["c"], {}, name="floor_node")
    rule = get_vjp_rule("Floor")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "ConstantOfShape"


def test_ceil_vjp() -> None:
    node = Node("Ceil", ["a"], ["c"], {}, name="ceil_node")
    rule = get_vjp_rule("Ceil")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "ConstantOfShape"


def test_equal_vjp() -> None:
    node = Node("Equal", ["a", "b"], ["c"], {}, name="equal_node")
    rule = get_vjp_rule("Equal")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2
    assert nodes[0].op_type == "ConstantOfShape"


def test_less_vjp() -> None:
    node = Node("Less", ["a", "b"], ["c"], {}, name="less_node")
    rule = get_vjp_rule("Less")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2


def test_greater_vjp() -> None:
    node = Node("Greater", ["a", "b"], ["c"], {}, name="greater_node")
    rule = get_vjp_rule("Greater")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2


def test_celu_vjp() -> None:
    node = Node("Celu", ["a"], ["c"], {}, name="celu_node")
    rule = get_vjp_rule("Celu")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "CeluGrad"


def test_mish_vjp() -> None:
    node = Node("Mish", ["a"], ["c"], {}, name="mish_node")
    rule = get_vjp_rule("Mish")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "MishGrad"


def test_shrink_vjp() -> None:
    node = Node("Shrink", ["a"], ["c"], {}, name="shrink_node")
    rule = get_vjp_rule("Shrink")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "ShrinkGrad"


def test_topk_vjp() -> None:
    node = Node("TopK", ["a", "k"], ["v", "i"], {}, name="topk_node")
    rule = get_vjp_rule("TopK")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_v"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "TopKGrad"


def test_spacetodepth_vjp() -> None:
    node = Node("SpaceToDepth", ["a"], ["c"], {}, name="spacetodepth_node")
    rule = get_vjp_rule("SpaceToDepth")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "DepthToSpace"


def test_depthtospace_vjp() -> None:
    node = Node("DepthToSpace", ["a"], ["c"], {}, name="depthtospace_node")
    rule = get_vjp_rule("DepthToSpace")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "SpaceToDepth"


def test_cumsum_vjp() -> None:
    node = Node("CumSum", ["a", "axis"], ["c"], {}, name="cumsum_node")
    rule = get_vjp_rule("CumSum")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "CumSum"


def test_reversesequence_vjp() -> None:
    node = Node("ReverseSequence", ["a", "lens"], ["c"], {}, name="reversesequence_node")
    rule = get_vjp_rule("ReverseSequence")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "ReverseSequence"


def test_compress_vjp() -> None:
    node = Node("Compress", ["a", "cond"], ["c"], {}, name="compress_node")
    rule = get_vjp_rule("Compress")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "CompressGrad"


def test_trilu_vjp() -> None:
    node = Node("Trilu", ["a", "k"], ["c"], {}, name="trilu_node")
    rule = get_vjp_rule("Trilu")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "Trilu"


def test_lpnormalization_vjp() -> None:
    node = Node("LpNormalization", ["a"], ["c"], {}, name="lpnorm_node")
    rule = get_vjp_rule("LpNormalization")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "LpNormalizationGrad"


def test_globallppool_vjp() -> None:
    node = Node("GlobalLpPool", ["a"], ["c"], {}, name="globallppool_node")
    rule = get_vjp_rule("GlobalLpPool")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "GlobalLpPoolGrad"


def test_einsum_vjp() -> None:
    node = Node("Einsum", ["a", "b"], ["c"], {}, name="einsum_node")
    rule = get_vjp_rule("Einsum")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2
    assert nodes[0].op_type == "EinsumGrad"


def test_maxroipool_vjp() -> None:
    node = Node("MaxRoiPool", ["a", "rois"], ["c"], {}, name="maxroipool_node")
    rule = get_vjp_rule("MaxRoiPool")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "MaxRoiPoolGrad"


def test_roialign_vjp() -> None:
    node = Node("RoiAlign", ["a", "rois", "batch"], ["c"], {}, name="roialign_node")
    rule = get_vjp_rule("RoiAlign")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "RoiAlignGrad"


def test_spacetobatchnd_vjp() -> None:
    node = Node("SpaceToBatchND", ["a", "bs", "p"], ["c"], {}, name="spacetobatchnd_node")
    rule = get_vjp_rule("SpaceToBatchND")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "BatchToSpaceND"


def test_batchtospacend_vjp() -> None:
    node = Node("BatchToSpaceND", ["a", "bs", "p"], ["c"], {}, name="batchtospacend_node")
    rule = get_vjp_rule("BatchToSpaceND")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 1
    assert nodes[0].op_type == "SpaceToBatchND"


def test_bitshift_vjp() -> None:
    node = Node("BitShift", ["a", "b"], ["c"], {}, name="bitshift_node")
    rule = get_vjp_rule("BitShift")
    (nodes, names) = rule.build_backward_nodes(node, ["grad_c"])
    assert len(nodes) == 2
    assert nodes[0].op_type == "ConstantOfShape"
