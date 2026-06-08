from typing import Any

import pytest
from onnx9000.jax.jax_ops import (
    _map_jax_add_prim,
    _map_jax_broadcast_in_dim_prim,
    _map_jax_cond,
    _map_jax_conv_general_dilated,
    _map_jax_div_prim,
    _map_jax_dot_general_prim,
    _map_jax_dynamic_slice,
    _map_jax_gather,
    _map_jax_grad_core_prim,
    _map_jax_mul_prim,
    _map_jax_pad,
    _map_jax_reduce_max,
    _map_jax_reduce_min,
    _map_jax_reduce_prod,
    _map_jax_reduce_sum,
    _map_jax_reduce_window_max,
    _map_jax_reduce_window_sum,
    _map_jax_scan,
    _map_jax_scatter,
    _map_jax_slice,
    _map_jax_sub_prim,
    _map_jax_while_loop,
    _map_jax_xla_pmap_prim,
)


def test_jax_ops():
    assert _map_jax_add_prim([], [], {}).op_type == "Add"
    assert _map_jax_add_prim(["a"], ["b"], {}).name == "add_b"
    assert _map_jax_mul_prim([], [], {}).op_type == "Mul"
    assert _map_jax_mul_prim(["a"], ["b"], {}).name == "mul_b"
    assert _map_jax_dot_general_prim([], [], {}).op_type == "MatMul"
    assert _map_jax_dot_general_prim(["a"], ["b"], {}).name == "dot_general_b"
    assert _map_jax_broadcast_in_dim_prim([], [], {}).op_type == "Expand"
    assert _map_jax_broadcast_in_dim_prim(["a"], ["b"], {}).name == "broadcast_in_dim_b"
    assert _map_jax_xla_pmap_prim([], [], {}).op_type == "XlaPmap"
    assert _map_jax_xla_pmap_prim(["a"], ["b"], {}).name == "xla_pmap_b"
    assert _map_jax_grad_core_prim([], [], {}).op_type == "Grad"
    assert _map_jax_grad_core_prim(["a"], ["b"], {}).name == "grad_core_b"
    assert _map_jax_sub_prim([], [], {}).op_type == "Sub"
    assert _map_jax_sub_prim(["a"], ["b"], {}).name == "sub_b"
    assert _map_jax_div_prim([], [], {}).op_type == "Div"
    assert _map_jax_div_prim(["a"], ["b"], {}).name == "div_b"

    # Conv
    conv_params = {
        "dimension_numbers": "1",
        "window_strides": [1, 1],
        "padding": [0, 0],
        "lhs_dilation": [1],
        "rhs_dilation": [1],
        "feature_group_count": 1,
    }
    conv_node = _map_jax_conv_general_dilated(["a"], ["b"], conv_params)
    assert conv_node.op_type == "Conv"
    assert conv_node.name == "conv_general_dilated_b"
    assert _map_jax_conv_general_dilated([], [], {}).name == "conv_general_dilated"

    assert _map_jax_reduce_sum([], [], {}).op_type == "ReduceSum"
    assert _map_jax_reduce_sum(["a"], ["b"], {}).name == "reduce_sum_b"
    assert _map_jax_reduce_max([], [], {}).op_type == "ReduceMax"
    assert _map_jax_reduce_max(["a"], ["b"], {}).name == "reduce_max_b"
    assert _map_jax_reduce_min([], [], {}).op_type == "ReduceMin"
    assert _map_jax_reduce_min(["a"], ["b"], {}).name == "reduce_min_b"
    assert _map_jax_reduce_prod([], [], {}).op_type == "ReduceProd"
    assert _map_jax_reduce_prod(["a"], ["b"], {}).name == "reduce_prod_b"

    assert _map_jax_reduce_window_max([], [], {}).op_type == "MaxPool"
    assert _map_jax_reduce_window_max(["a"], ["b"], {}).name == "reduce_window_max_b"
    assert _map_jax_reduce_window_sum([], [], {}).op_type == "AveragePool"
    assert _map_jax_reduce_window_sum(["a"], ["b"], {}).name == "reduce_window_sum_b"

    assert _map_jax_pad([], [], {}).op_type == "Pad"
    assert _map_jax_pad(["a"], ["b"], {}).name == "pad_b"
    assert _map_jax_slice([], [], {}).op_type == "Slice"
    assert _map_jax_slice(["a"], ["b"], {}).name == "slice_b"
    assert _map_jax_dynamic_slice([], [], {}).op_type == "Slice"
    assert _map_jax_dynamic_slice(["a"], ["b"], {}).name == "dynamic_slice_b"
    assert _map_jax_gather([], [], {}).op_type == "GatherElements"
    assert _map_jax_gather(["a"], ["b"], {}).name == "gather_b"
    assert _map_jax_scatter([], [], {}).op_type == "ScatterND"
    assert _map_jax_scatter(["a"], ["b"], {}).name == "scatter_b"
    assert _map_jax_cond([], [], {}).op_type == "If"
    assert _map_jax_cond(["a"], ["b"], {}).name == "cond_b"
    assert _map_jax_scan([], [], {}).op_type == "Scan"
    assert _map_jax_scan(["a"], ["b"], {}).name == "scan_b"
    assert _map_jax_while_loop([], [], {}).op_type == "Loop"
    assert _map_jax_while_loop(["a"], ["b"], {}).name == "while_loop_b"


from onnx9000.jax.jaxpr_string_parser import parse_jaxpr_string


def test_jaxpr_string_parser():
    test_str = """
{
  in (a, b)
  out = op[attr=1, attr2='test'] input1 input2
  out2 = op2[attr3=(1,2)] input3
  out3 = op3[attr4=invalid_eval] input4
}
    """

    res = parse_jaxpr_string(test_str)
    assert len(res["outvars"]) == 2
    assert res["outvars"][0]["name"] == "a"
    assert res["outvars"][1]["name"] == "b"

    # For coverage of line 55: Exception on eval
    res2 = parse_jaxpr_string("out = op[attr=definitely_not_evalable] i1")
    assert res2["eqns"][0]["params"]["attr"] == "definitely_not_evalable"
    assert len(res["eqns"]) == 3
    assert res["eqns"][0]["primitive"] == "op"
    assert res["eqns"][0]["invars"] == [{"name": "input1"}, {"name": "input2"}]
    assert res["eqns"][0]["params"]["attr"] == 1
    assert res["eqns"][0]["params"]["attr2"] == "test"

    assert res["eqns"][1]["params"]["attr3"] == (1, 2)
    assert res["eqns"][2]["params"]["attr4"] == "invalid_eval"

    # Empty
    assert parse_jaxpr_string("") == {"invars": [], "outvars": [], "constvars": [], "eqns": []}
