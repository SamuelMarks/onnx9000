"""Pooling and Reduction operation implementations for ONNX to C89 generation."""

from onnx9000.c_compiler.ast_builder import C89Builder
from onnx9000.c_compiler.spatial import get_attribute
from onnx9000.core.ir import Node, Tensor
from onnx9000.core.profiler import resolve_volume


def generate_pooling(
    b: C89Builder,
    node: Node,
    out_tensor: Tensor,
    in_tensor: Tensor,
    in_name: str,
    out_name: str,
    pool_type: str,
):
    """Generate MaxPool and AveragePool loops (1D, 2D)."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()

    in_shape = in_tensor.shape
    out_shape = out_tensor.shape
    spatial_dims = len(in_shape) - 2

    kernel_shape = get_attribute(node, "kernel_shape", [1] * spatial_dims)
    strides = get_attribute(node, "strides", [1] * spatial_dims)
    pads = get_attribute(node, "pads", [0] * (spatial_dims * 2))

    batch = in_shape[0]
    channels = in_shape[1]

    is_max = "Max" in pool_type
    init_val = (
        "-1e38f" if is_max else "0.0f"
    )  # Using -1e38f instead of -FLT_MAX to avoid float.h include complexity

    if spatial_dims == 2:
        IH, IW = in_shape[2], in_shape[3]
        OH, OW = out_shape[2], out_shape[3]
        KH, KW = kernel_shape[0], kernel_shape[1]
        SH, SW = strides[0], strides[1]
        PT, PL = pads[0], pads[1]

        b.emit("int b_idx, c, oh, ow, kh, kw;")
        b.emit(f"for (b_idx = 0; b_idx < {batch}; ++b_idx) {{")
        b.push_indent()
        b.emit(f"for (c = 0; c < {channels}; ++c) {{")
        b.push_indent()
        b.emit(f"for (oh = 0; oh < {OH}; ++oh) {{")
        b.push_indent()
        b.emit(f"for (ow = 0; ow < {OW}; ++ow) {{")
        b.push_indent()

        b.emit(f"float val = {init_val};")
        if not is_max:
            b.emit("int count = 0;")

        b.emit(f"for (kh = 0; kh < {KH}; ++kh) {{")
        b.push_indent()
        b.emit(f"for (kw = 0; kw < {KW}; ++kw) {{")
        b.push_indent()
        b.emit(f"int ih = oh * {SH} + kh - {PT};")
        b.emit(f"int iw = ow * {SW} + kw - {PL};")
        b.emit(f"if (ih >= 0 && ih < {IH} && iw >= 0 && iw < {IW}) {{")
        b.push_indent()
        b.emit(
            f"float current = {in_name}[b_idx * {channels * IH * IW} + c * {IH * IW} + ih * {IW} + iw];"
        )
        if is_max:
            b.emit("if (current > val) val = current;")
        else:
            b.emit("val += current;")
            b.emit("count++;")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")

        if not is_max:
            b.emit("if (count > 0) val /= count;")

        b.emit(f"{out_name}[b_idx * {channels * OH * OW} + c * {OH * OW} + oh * {OW} + ow] = val;")

        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")

    elif spatial_dims == 1:
        IW = in_shape[2]
        OW = out_shape[2]
        KW = kernel_shape[0]
        SW = strides[0]
        PL = pads[0]

        b.emit("int b_idx, c, ow, kw;")
        b.emit(f"for (b_idx = 0; b_idx < {batch}; ++b_idx) {{")
        b.push_indent()
        b.emit(f"for (c = 0; c < {channels}; ++c) {{")
        b.push_indent()
        b.emit(f"for (ow = 0; ow < {OW}; ++ow) {{")
        b.push_indent()
        b.emit(f"float val = {init_val};")
        if not is_max:
            b.emit("int count = 0;")

        b.emit(f"for (kw = 0; kw < {KW}; ++kw) {{")
        b.push_indent()
        b.emit(f"int iw = ow * {SW} + kw - {PL};")
        b.emit(f"if (iw >= 0 && iw < {IW}) {{")
        b.push_indent()
        b.emit(f"float current = {in_name}[b_idx * {channels * IW} + c * {IW} + iw];")
        if is_max:
            b.emit("if (current > val) val = current;")
        else:
            b.emit("val += current;")
            b.emit("count++;")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")

        if not is_max:
            b.emit("if (count > 0) val /= count;")

        b.emit(f"{out_name}[b_idx * {channels * OW} + c * {OW} + ow] = val;")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
    else:
        b.emit("/* Unsupported Pooling dim */")

    b.pop_indent()
    b.emit("}")


def generate_global_pooling(
    b: C89Builder,
    node: Node,
    out_tensor: Tensor,
    in_tensor: Tensor,
    in_name: str,
    out_name: str,
    pool_type: str,
):
    """Generate GlobalMaxPool and GlobalAveragePool loops."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()

    in_shape = in_tensor.shape
    batch = in_shape[0]
    channels = in_shape[1]

    spatial_volume = 1
    for dim in in_shape[2:]:
        spatial_volume *= dim

    is_max = "Max" in pool_type
    init_val = "-1e38f" if is_max else "0.0f"

    b.emit("int b_idx, c, s;")
    b.emit(f"for (b_idx = 0; b_idx < {batch}; ++b_idx) {{")
    b.push_indent()
    b.emit(f"for (c = 0; c < {channels}; ++c) {{")
    b.push_indent()
    b.emit(f"float val = {init_val};")

    b.emit(f"for (s = 0; s < {spatial_volume}; ++s) {{")
    b.push_indent()
    b.emit(
        f"float current = {in_name}[b_idx * {channels * spatial_volume} + c * {spatial_volume} + s];"
    )
    if is_max:
        b.emit("if (current > val) val = current;")
    else:
        b.emit("val += current;")
    b.pop_indent()
    b.emit("}")

    if not is_max:
        b.emit(f"val /= {spatial_volume};")

    b.emit(f"{out_name}[b_idx * {channels} + c] = val;")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")

    b.pop_indent()
    b.emit("}")


def generate_reduction(
    b: C89Builder,
    node: Node,
    out_tensor: Tensor,
    in_tensor: Tensor,
    in_name: str,
    out_name: str,
    reduce_op: str,
):
    """Generate ReduceMean, ReduceSum, ReduceMax, ReduceMin, ReduceProd loops."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()

    in_shape = in_tensor.shape
    axes = get_attribute(node, "axes", list(range(len(in_shape))))
    get_attribute(node, "keepdims", 1)

    # Simple continuous flat array reduction if axes cover all dimensions
    if len(axes) == len(in_shape):
        b.emit("/* Optimized Flat Reduction */")
        volume = resolve_volume(in_shape)
        init_val = "0.0f"
        if reduce_op == "Max":
            init_val = "-1e38f"
        elif reduce_op == "Min":
            init_val = "1e38f"
        elif reduce_op == "Prod":
            init_val = "1.0f"

        b.emit(f"float val = {init_val};")
        b.emit("int i;")
        b.emit(f"for (i = 0; i < {volume}; ++i) {{")
        b.push_indent()
        b.emit(f"float current = {in_name}[i];")
        if reduce_op in ["Sum", "Mean"]:
            b.emit("val += current;")
        elif reduce_op == "Max":
            b.emit("if (current > val) val = current;")
        elif reduce_op == "Min":
            b.emit("if (current < val) val = current;")
        elif reduce_op == "Prod":
            b.emit("val *= current;")
        b.pop_indent()
        b.emit("}")

        if reduce_op == "Mean":
            b.emit(f"val /= {volume};")

        b.emit(f"{out_name}[0] = val;")
    else:
        pass

    b.pop_indent()
    b.emit("}")


def generate_arg_reduction(
    b: C89Builder,
    node: Node,
    out_tensor: Tensor,
    in_tensor: Tensor,
    in_name: str,
    out_name: str,
    reduce_op: str,
):
    """Generate ArgMax, ArgMin loops using int64_t."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()

    in_shape = in_tensor.shape
    axis = get_attribute(node, "axis", 0)
    get_attribute(node, "keepdims", 1)
    select_last_index = get_attribute(node, "select_last_index", 0)

    is_max = "Max" in reduce_op

    b.emit("/* Arg Reduction natively emitting to int64_t / int32_t indices */")

    pre_axis_vol = resolve_volume(in_shape[:axis]) if axis > 0 else 1
    axis_dim = in_shape[axis]
    post_axis_vol = resolve_volume(in_shape[axis + 1 :]) if axis < len(in_shape) - 1 else 1

    init_val = "-1e38f" if is_max else "1e38f"

    b.emit("int pre, post, d;")
    b.emit(f"for (pre = 0; pre < {pre_axis_vol}; ++pre) {{")
    b.push_indent()
    b.emit(f"for (post = 0; post < {post_axis_vol}; ++post) {{")
    b.push_indent()
    b.emit(f"float best_val = {init_val};")
    b.emit("int64_t best_idx = 0;")

    b.emit(f"for (d = 0; d < {axis_dim}; ++d) {{")
    b.push_indent()
    b.emit(
        f"float current = {in_name}[pre * {axis_dim * post_axis_vol} + d * {post_axis_vol} + post];"
    )

    if is_max:
        b.emit(
            "if (current > best_val"
            + (" >= " if select_last_index else " > ")
            + "best_val) { best_val = current; best_idx = d; }"
        )
    else:
        b.emit(
            "if (current < best_val"
            + (" <= " if select_last_index else " < ")
            + "best_val) { best_val = current; best_idx = d; }"
        )

    b.pop_indent()
    b.emit("}")

    b.emit(f"{out_name}[pre * {post_axis_vol} + post] = best_idx;")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")

    b.pop_indent()
    b.emit("}")
