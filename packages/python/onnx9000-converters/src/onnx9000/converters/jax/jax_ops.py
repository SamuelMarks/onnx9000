"""Module providing core logic and structural definitions for jax ops."""

from typing import Any

from onnx9000.core.ir import Node
from onnx9000.core.registry import register_op


@register_op("add", "jax")
def _map_jax_add_prim(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_add_prim operation."""
    return Node(
        op_type="Add",
        inputs=inputs,
        outputs=outputs,
        name=f"add_{outputs[0]}" if outputs else "add",
    )


@register_op("mul", "jax")
def _map_jax_mul_prim(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_mul_prim operation."""
    return Node(
        op_type="Mul",
        inputs=inputs,
        outputs=outputs,
        name=f"mul_{outputs[0]}" if outputs else "mul",
    )


@register_op("dot_general", "jax")
def _map_jax_dot_general_prim(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_dot_general_prim operation."""
    return Node(
        op_type="MatMul",
        inputs=inputs,
        outputs=outputs,
        name=f"dot_general_{outputs[0]}" if outputs else "dot_general",
    )


@register_op("broadcast_in_dim", "jax")
def _map_jax_broadcast_in_dim_prim(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """Execute the _map_jax_broadcast_in_dim_prim operation."""
    return Node(
        op_type="Expand",
        inputs=inputs,
        outputs=outputs,
        attributes={"broadcast_dimensions": params.get("broadcast_dimensions", [])},
        name=f"broadcast_in_dim_{outputs[0]}" if outputs else "broadcast_in_dim",
    )


@register_op("xla_pmap", "jax")
def _map_jax_xla_pmap_prim(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_xla_pmap_prim operation."""
    return Node(
        op_type="XlaPmap",
        inputs=inputs,
        outputs=outputs,
        attributes={"axis_name": params.get("axis_name", "")},
        name=f"xla_pmap_{outputs[0]}" if outputs else "xla_pmap",
    )


@register_op("grad_core", "jax")
def _map_jax_grad_core_prim(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """Execute the _map_jax_grad_core_prim operation."""
    return Node(
        op_type="Grad",
        inputs=inputs,
        outputs=outputs,
        name=f"grad_core_{outputs[0]}" if outputs else "grad_core",
    )


@register_op("sub", "jax")
def _map_jax_sub_prim(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax sub prim."""
    return Node(
        op_type="Sub",
        inputs=inputs,
        outputs=outputs,
        name=f"sub_{outputs[0]}" if outputs else "sub",
    )


@register_op("div", "jax")
def _map_jax_div_prim(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax div prim."""
    return Node(
        op_type="Div",
        inputs=inputs,
        outputs=outputs,
        name=f"div_{outputs[0]}" if outputs else "div",
    )


@register_op("conv_general_dilated", "jax")
def _map_jax_conv_general_dilated(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    # Need to handle RHS and LHS dilation and dimension mapping properly,
    # mapping this strictly to IR.ConvND according to the plan.
    """map jax conv general dilated."""
    attributes = {}
    if "dimension_numbers" in params:
        # In a real implementation we'd translate these to ONNX formats
        attributes["dimension_numbers"] = str(params["dimension_numbers"])
    if "window_strides" in params:
        attributes["strides"] = list(params["window_strides"])
    if "padding" in params:
        attributes["pads"] = list(params["padding"])
    if "lhs_dilation" in params:
        attributes["lhs_dilation"] = list(params["lhs_dilation"])
    if "rhs_dilation" in params:
        attributes["dilations"] = list(params["rhs_dilation"])
    if "feature_group_count" in params:
        attributes["group"] = params["feature_group_count"]

    return Node(
        op_type="Conv",
        inputs=inputs,
        outputs=outputs,
        attributes=attributes,
        name=f"conv_general_dilated_{outputs[0]}" if outputs else "conv_general_dilated",
    )


@register_op("reduce_sum", "jax")
def _map_jax_reduce_sum(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax reduce sum."""
    return Node(
        op_type="ReduceSum",
        inputs=inputs,
        outputs=outputs,
        attributes={"axes": params.get("axes", [])},
        name=f"reduce_sum_{outputs[0]}" if outputs else "reduce_sum",
    )


@register_op("reduce_max", "jax")
def _map_jax_reduce_max(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax reduce max."""
    return Node(
        op_type="ReduceMax",
        inputs=inputs,
        outputs=outputs,
        attributes={"axes": params.get("axes", [])},
        name=f"reduce_max_{outputs[0]}" if outputs else "reduce_max",
    )


@register_op("reduce_min", "jax")
def _map_jax_reduce_min(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax reduce min."""
    return Node(
        op_type="ReduceMin",
        inputs=inputs,
        outputs=outputs,
        attributes={"axes": params.get("axes", [])},
        name=f"reduce_min_{outputs[0]}" if outputs else "reduce_min",
    )


@register_op("reduce_prod", "jax")
def _map_jax_reduce_prod(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax reduce prod."""
    return Node(
        op_type="ReduceProd",
        inputs=inputs,
        outputs=outputs,
        attributes={"axes": params.get("axes", [])},
        name=f"reduce_prod_{outputs[0]}" if outputs else "reduce_prod",
    )


@register_op("reduce_window_max", "jax")
def _map_jax_reduce_window_max(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """map jax reduce window max."""
    return Node(
        op_type="MaxPool",
        inputs=inputs,
        outputs=outputs,
        attributes={
            "kernel_shape": params.get("window_dimensions", []),
            "strides": params.get("window_strides", []),
            "pads": params.get("padding", []),
        },
        name=f"reduce_window_max_{outputs[0]}" if outputs else "reduce_window_max",
    )


@register_op("reduce_window_sum", "jax")
def _map_jax_reduce_window_sum(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """map jax reduce window sum."""
    return Node(
        op_type="AveragePool",
        inputs=inputs,
        outputs=outputs,
        attributes={
            "kernel_shape": params.get("window_dimensions", []),
            "strides": params.get("window_strides", []),
            "pads": params.get("padding", []),
        },
        name=f"reduce_window_sum_{outputs[0]}" if outputs else "reduce_window_sum",
    )


@register_op("pad", "jax")
def _map_jax_pad(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax pad."""
    return Node(
        op_type="Pad",
        inputs=inputs,
        outputs=outputs,
        attributes={"padding_config": str(params.get("padding_config", []))},
        name=f"pad_{outputs[0]}" if outputs else "pad",
    )


@register_op("slice", "jax")
def _map_jax_slice(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax slice."""
    return Node(
        op_type="Slice",
        inputs=inputs,
        outputs=outputs,
        attributes={
            "start_indices": params.get("start_indices", []),
            "limit_indices": params.get("limit_indices", []),
            "strides": params.get("strides", []),
        },
        name=f"slice_{outputs[0]}" if outputs else "slice",
    )


@register_op("dynamic_slice", "jax")
def _map_jax_dynamic_slice(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax dynamic slice."""
    return Node(
        op_type="Slice",
        inputs=inputs,
        outputs=outputs,
        attributes={"slice_sizes": params.get("slice_sizes", [])},
        name=f"dynamic_slice_{outputs[0]}" if outputs else "dynamic_slice",
    )


@register_op("gather", "jax")
def _map_jax_gather(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax gather."""
    return Node(
        op_type="GatherElements",
        inputs=inputs,
        outputs=outputs,
        attributes={"dimension_numbers": str(params.get("dimension_numbers", {}))},
        name=f"gather_{outputs[0]}" if outputs else "gather",
    )


@register_op("scatter", "jax")
def _map_jax_scatter(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax scatter."""
    return Node(
        op_type="ScatterND",
        inputs=inputs,
        outputs=outputs,
        attributes={"dimension_numbers": str(params.get("dimension_numbers", {}))},
        name=f"scatter_{outputs[0]}" if outputs else "scatter",
    )


@register_op("cond", "jax")
def _map_jax_cond(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax cond."""
    return Node(
        op_type="If",
        inputs=inputs,
        outputs=outputs,
        attributes={
            "true_branch": str(params.get("true_jaxpr", {})),
            "false_branch": str(params.get("false_jaxpr", {})),
        },
        name=f"cond_{outputs[0]}" if outputs else "cond",
    )


@register_op("scan", "jax")
def _map_jax_scan(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax scan."""
    return Node(
        op_type="Scan",
        inputs=inputs,
        outputs=outputs,
        attributes={"body": str(params.get("jaxpr", {}))},
        name=f"scan_{outputs[0]}" if outputs else "scan",
    )


@register_op("while_loop", "jax")
def _map_jax_while_loop(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map jax while loop."""
    return Node(
        op_type="Loop",
        inputs=inputs,
        outputs=outputs,
        attributes={
            "cond": str(params.get("cond_jaxpr", {})),
            "body": str(params.get("body_jaxpr", {})),
        },
        name=f"while_loop_{outputs[0]}" if outputs else "while_loop",
    )
