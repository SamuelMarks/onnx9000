"""Module docstring."""

from typing import Callable

from onnx9000.converters.paddle.builder import PaddleToONNXGraphBuilder
from onnx9000.converters.paddle.parsers import PaddleNode


def _map_reshape(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map reshape operation."""
    inputs = []
    if "X" in node.inputs:
        inputs.extend(node.inputs["X"])
    if "Shape" in node.inputs:
        inputs.extend(node.inputs["Shape"])
    if "Shape" not in node.inputs:
        shape_attr = builder.extract_list_attr(node, "shape")
        shape_tensor = builder.add_constant(
            f"{node.name}_shape_attr", shape_attr, 7, (len(shape_attr),)
        )
        inputs.append(shape_tensor)
    out_names = node.outputs.get("Out", [])
    if not out_names:
        out_names = [f"{node.name}_out"]
    builder.make_node("Reshape", inputs, {}, node.name, outputs=out_names)
    return out_names


def _map_transpose(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map transpose operation."""
    inputs = node.inputs.get("X", [])
    perm = builder.extract_list_attr(node, "axis")
    out_names = node.outputs.get("Out", [])
    if not out_names:
        out_names = [f"{node.name}_out"]
    builder.make_node("Transpose", inputs, {"perm": perm}, node.name, outputs=out_names)
    return out_names


def _map_flatten(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map flatten operation."""
    inputs = node.inputs.get("X", [])
    axis = builder.extract_attr(node, "axis", 1)
    out_names = node.outputs.get("Out", [])
    if not out_names:
        out_names = [f"{node.name}_out"]
    builder.make_node("Flatten", inputs, {"axis": axis}, node.name, outputs=out_names)
    return out_names


def _map_squeeze(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map squeeze operation."""
    inputs = node.inputs.get("X", [])
    axes = builder.extract_list_attr(node, "axes")
    out_names = node.outputs.get("Out", [])
    if not out_names:
        out_names = [f"{node.name}_out"]
    if axes:
        axes_tensor = builder.add_constant(f"{node.name}_axes", axes, 7, (len(axes),))
        inputs.append(axes_tensor)
    builder.make_node("Squeeze", inputs, {}, node.name, outputs=out_names)
    return out_names


def _map_unsqueeze(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map unsqueeze operation."""
    inputs = node.inputs.get("X", [])
    axes = builder.extract_list_attr(node, "axes")
    out_names = node.outputs.get("Out", [])
    if not out_names:
        out_names = [f"{node.name}_out"]
    if axes:
        axes_tensor = builder.add_constant(f"{node.name}_axes", axes, 7, (len(axes),))
        inputs.append(axes_tensor)
    builder.make_node("Unsqueeze", inputs, {}, node.name, outputs=out_names)
    return out_names


def _map_concat(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map concat operation."""
    inputs = node.inputs.get("X", [])
    axis = builder.extract_attr(node, "axis", 0)
    return builder.make_node("Concat", inputs, {"axis": axis}, node.name)


def _map_stack(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map stack operation."""
    inputs = node.inputs.get("X", [])
    axis = builder.extract_attr(node, "axis", 0)
    unsqueezed = []
    axes_const = builder.add_constant(f"{node.name}_axes", [axis], 7, (1,))
    for i, inp in enumerate(inputs):
        u = builder.make_node("Unsqueeze", [inp, axes_const], {}, f"{node.name}_unsq_{i}")[0]
        unsqueezed.append(u)
    return builder.make_node("Concat", unsqueezed, {"axis": axis}, node.name)


def _map_unstack(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map unstack operation."""
    inputs = node.inputs.get("X", [])
    axis = builder.extract_attr(node, "axis", 0)
    num = builder.extract_attr(node, "num", len(node.outputs.get("Y", [])))
    splits = builder.make_node(
        "Split", inputs, {"axis": axis, "num_outputs": num}, f"{node.name}_split"
    )
    axes_const = builder.add_constant(f"{node.name}_axes", [axis], 7, (1,))
    outs = []
    for i, s in enumerate(splits):
        sq = builder.make_node("Squeeze", [s, axes_const], {}, f"{node.name}_sq_{i}")[0]
        outs.append(sq)
    expected_outs = node.outputs.get("Y", [])
    for i, out_name in enumerate(expected_outs):
        if i < len(outs):
            builder.rewire_edge(outs[i], out_name)
    return expected_outs if expected_outs else outs


def _map_split(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map split operation."""
    inputs = node.inputs.get("X", [])
    num = builder.extract_attr(node, "num", 1)
    axis = builder.extract_attr(node, "axis", 0)
    out_names = node.outputs.get("Out", [])
    if not out_names:
        out_names = [f"{node.name}_out_{i}" for i in range(num)]
    builder.make_node(
        "Split", inputs, {"axis": axis, "num_outputs": num}, node.name, outputs=out_names
    )
    return out_names


def _map_slice(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map slice operation."""
    inputs = node.inputs.get("X", [])
    if not inputs:
        inputs = node.inputs.get("Input", [])
    axes = builder.extract_list_attr(node, "axes")
    starts = builder.extract_list_attr(node, "starts")
    ends = builder.extract_list_attr(node, "ends")
    starts_const = builder.add_constant(f"{node.name}_starts", starts, 7, [len(starts)])
    ends_const = builder.add_constant(f"{node.name}_ends", ends, 7, [len(ends)])
    axes_const = builder.add_constant(f"{node.name}_axes", axes, 7, [len(axes)])
    slice_out = builder.make_node(
        "Slice", inputs + [starts_const, ends_const, axes_const], {}, f"{node.name}_slice"
    )[0]
    decrease_axis = builder.extract_list_attr(node, "decrease_axis")
    if decrease_axis:
        da_const = builder.add_constant(f"{node.name}_da", decrease_axis, 7, [len(decrease_axis)])
        return builder.make_node("Squeeze", [slice_out, da_const], {}, node.name)
    return builder.make_node("Identity", [slice_out], {}, node.name)


def _map_gather(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map gather operation."""
    inputs = []
    if "X" in node.inputs:
        inputs.extend(node.inputs["X"])
    if "Index" in node.inputs:
        inputs.extend(node.inputs["Index"])
    axis = builder.extract_attr(node, "axis", 0)
    return builder.make_node("Gather", inputs, {"axis": axis}, node.name)


def _map_gather_nd(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map gather nd operation."""
    inputs = []
    if "X" in node.inputs:
        inputs.extend(node.inputs["X"])
    if "Index" in node.inputs:
        inputs.extend(node.inputs["Index"])
    return builder.make_node("GatherND", inputs, {}, node.name)


def _map_scatter(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map scatter operation."""
    inputs = []
    if "X" in node.inputs:
        inputs.extend(node.inputs["X"])
    if "Index" in node.inputs:
        inputs.extend(node.inputs["Index"])
    if "Updates" in node.inputs:
        inputs.extend(node.inputs["Updates"])
    return builder.make_node("ScatterElements", inputs, {}, node.name)


def _map_scatter_nd(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map scatter nd operation."""
    inputs = []
    if "X" in node.inputs:
        inputs.extend(node.inputs["X"])
    if "Index" in node.inputs:
        inputs.extend(node.inputs["Index"])
    if "Updates" in node.inputs:
        inputs.extend(node.inputs["Updates"])
    return builder.make_node("ScatterND", inputs, {}, node.name)


def _map_scatter_nd_add(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map scatter nd add operation."""
    inputs = []
    if "X" in node.inputs:
        inputs.extend(node.inputs["X"])
    if "Index" in node.inputs:
        inputs.extend(node.inputs["Index"])
    if "Updates" in node.inputs:
        inputs.extend(node.inputs["Updates"])
    return builder.make_node("ScatterND", inputs, {"reduction": "add"}, node.name)


def _map_tile(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map tile operation."""
    inputs = []
    if "X" in node.inputs:
        inputs.extend(node.inputs["X"])
    if "RepeatTimes" in node.inputs:
        inputs.extend(node.inputs["RepeatTimes"])
    return builder.make_node("Tile", inputs, {}, node.name)


def _map_expand(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map expand operation."""
    inputs = []
    if "X" in node.inputs:
        inputs.extend(node.inputs["X"])
    if "Shape" in node.inputs:
        inputs.extend(node.inputs["Shape"])
    return builder.make_node("Expand", inputs, {}, node.name)


def _map_expand_as(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map expand as operation."""
    x = node.inputs.get("X", [])
    y = node.inputs.get("Y", [])
    y_shape = builder.make_node("Shape", y, {}, f"{node.name}_shape")[0]
    return builder.make_node("Expand", x + [y_shape], {}, node.name)


def _map_cast(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map cast operation."""
    inputs = node.inputs.get("X", [])
    builder.extract_attr(node, "out_dtype", 5)
    return builder.make_node("Cast", inputs, {"to": 1}, node.name)


def _map_shape(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map shape operation."""
    return builder.make_node("Shape", node.inputs.get("Input", []), {}, node.name)


def _map_size(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map size operation."""
    return builder.make_node("Size", node.inputs.get("Input", []), {}, node.name)


def _map_fill_constant(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map fill constant operation."""
    shape = node.inputs.get("ShapeTensor", [])
    if not shape:
        shape_attr = builder.extract_list_attr(node, "shape")
        shape_const = builder.add_constant(f"{node.name}_shape", shape_attr, 7, (len(shape_attr),))
        shape.append(shape_const)
    val = builder.extract_attr(node, "value", 0.0)
    return builder.make_node("ConstantOfShape", shape, {"value": val}, node.name)


def _map_zeros_ones_like(val: float) -> Callable:
    """Executes the  map zeros ones like operation."""

    def _impl(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
        """Executes the  impl operation."""
        inputs = node.inputs.get("X", [])
        shape = builder.make_node("Shape", inputs, {}, f"{node.name}_shape")[0]
        return builder.make_node("ConstantOfShape", [shape], {"value": val}, node.name)

    return _impl


def _map_range(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map range operation."""
    inputs = []
    if "Start" in node.inputs:
        inputs.extend(node.inputs["Start"])
    if "End" in node.inputs:
        inputs.extend(node.inputs["End"])
    if "Step" in node.inputs:
        inputs.extend(node.inputs["Step"])
    return builder.make_node("Range", inputs, {}, node.name)


def _map_assign(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map assign operation."""
    inputs = node.inputs.get("X", [])
    if not inputs:
        val = builder.extract_attr(node, "values", [0.0])
        shape = builder.extract_list_attr(node, "shape")
        const = builder.add_constant(f"{node.name}_const", val, 1, tuple(shape))
        return builder.make_node("Identity", [const], {}, node.name)
    return builder.make_node("Identity", inputs, {}, node.name)


def _map_where(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map where operation."""
    inputs = []
    if "Condition" in node.inputs:
        inputs.extend(node.inputs["Condition"])
    if "X" in node.inputs:
        inputs.extend(node.inputs["X"])
    if "Y" in node.inputs:
        inputs.extend(node.inputs["Y"])
    return builder.make_node("Where", inputs, {}, node.name)


def _map_nonzero(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map nonzero operation."""
    inputs = node.inputs.get("Condition", [])
    return builder.make_node("NonZero", inputs, {}, node.name)


def _map_custom(op_name: str):

    def _impl(builder, node):
        inputs = node.inputs.get("X", [])
        if "Y" in node.inputs:
            inputs.extend(node.inputs["Y"])
        return builder.make_node(op_name, inputs, {}, node.name)

    return _impl


TENSOR_OPS_MAPPING: dict[str, Callable[[PaddleToONNXGraphBuilder, PaddleNode], list[str]]] = {
    "reshape": _map_reshape,
    "reshape2": _map_reshape,
    "transpose": _map_transpose,
    "transpose2": _map_transpose,
    "flatten": _map_flatten,
    "flatten_contiguous_range": _map_flatten,
    "squeeze": _map_squeeze,
    "squeeze2": _map_squeeze,
    "unsqueeze": _map_unsqueeze,
    "unsqueeze2": _map_unsqueeze,
    "concat": _map_concat,
    "stack": _map_stack,
    "unstack": _map_unstack,
    "split": _map_split,
    "slice": _map_slice,
    "strided_slice": _map_slice,
    "crop_tensor": _map_slice,
    "gather": _map_gather,
    "gather_nd": _map_gather_nd,
    "scatter": _map_scatter,
    "scatter_nd": _map_scatter_nd,
    "scatter_nd_add": _map_scatter_nd_add,
    "index_select": _map_gather,
    "tile": _map_tile,
    "expand": _map_expand,
    "expand_v2": _map_expand,
    "expand_as": _map_expand_as,
    "cast": _map_cast,
    "shape": _map_shape,
    "size": _map_size,
    "fill_constant": _map_fill_constant,
    "fill_constant_batch_size_like": _map_fill_constant,
    "fill_any_like": _map_fill_constant,
    "zeros_like": _map_zeros_ones_like(0.0),
    "ones_like": _map_zeros_ones_like(1.0),
    "arange": _map_range,
    "range": _map_range,
    "assign_value": _map_assign,
    "assign": _map_assign,
    "where": _map_where,
    "where_index": _map_nonzero,
    "nonzero": _map_nonzero,
    "expand_as_v2": _map_expand_as,
    "top_k": _map_custom("TopK"),
    "top_k_v2": _map_custom("TopK"),
    "argsort": _map_custom("TopK"),
    "unique": _map_custom("Unique"),
    "linspace": _map_custom("Custom_Paddle_linspace"),
    "eye": _map_custom("Custom_Paddle_eye"),
    "uniform_random": _map_custom("RandomUniform"),
    "gaussian_random": _map_custom("RandomNormal"),
    "randint": _map_custom("RandomUniform"),
    "randperm": _map_custom("Custom_Paddle_randperm"),
    "roll": _map_custom("Custom_Paddle_roll"),
    "flip": _map_custom("ReverseSequence"),
    "unbind": _map_unstack,
    "meshgrid": _map_custom("Custom_Paddle_meshgrid"),
    "bincount": _map_custom("Bincount"),
    "histogram": _map_custom("Histogram"),
    "kthvalue": _map_custom("TopK"),
    "mode": _map_custom("TopK"),
    "sort": _map_custom("TopK"),
    "diag": _map_custom("Custom_Paddle_diag"),
    "diag_embed": _map_custom("Custom_Paddle_diag_embed"),
    "trace": _map_custom("Custom_Paddle_trace"),
    "triu": _map_custom("Trilu"),
    "tril": _map_custom("Trilu"),
    "gru_unit": _map_custom("Custom_Paddle_gru_unit"),
    "lstm_unit": _map_custom("Custom_Paddle_lstm_unit"),
    "beam_search": _map_custom("Custom_Paddle_beam_search"),
    "beam_search_decode": _map_custom("Custom_Paddle_beam_search_decode"),
    "crf_decoding": _map_custom("Custom_Paddle_crf_decoding"),
    "viterbi_decode": _map_custom("Custom_Paddle_viterbi_decode"),
    "fused_attention": _map_custom("Custom_Paddle_fused_attention"),
    "fused_feedforward": _map_custom("Custom_Paddle_fused_feedforward"),
    "quantize_linear": _map_custom("QuantizeLinear"),
    "dequantize_linear": _map_custom("DequantizeLinear"),
    "fake_quantize_abs_max": _map_custom("Custom_Paddle_fake_quantize_abs_max"),
    "fake_quantize_range_abs_max": _map_custom("Custom_Paddle_fake_quantize_range_abs_max"),
    "fake_quantize_moving_average_abs_max": _map_custom(
        "Custom_Paddle_fake_quantize_moving_average_abs_max"
    ),
    "fake_channel_wise_quantize_abs_max": _map_custom(
        "Custom_Paddle_fake_channel_wise_quantize_abs_max"
    ),
    "fake_dequantize_max_abs": _map_custom("Custom_Paddle_fake_dequantize_max_abs"),
    "fake_channel_wise_dequantize_max_abs": _map_custom(
        "Custom_Paddle_fake_channel_wise_dequantize_max_abs"
    ),
}
