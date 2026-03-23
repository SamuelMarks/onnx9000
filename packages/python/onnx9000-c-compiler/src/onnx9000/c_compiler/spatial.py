"""Spatial operation implementations for ONNX to C89 generation."""

from onnx9000.c_compiler.ast_builder import C89Builder
from onnx9000.core.ir import Node, Tensor


def get_attribute(node, name, default):
    if node.attributes and name in node.attributes:
        val = node.attributes[name].value
        return val if val is not None else default
    return default


def generate_conv(
    b: C89Builder,
    node: Node,
    out_tensor: Tensor,
    in_tensor: Tensor,
    w_tensor: Tensor,
    bias_tensor: Tensor,
    in_name: str,
    w_name: str,
    bias_name: str,
    out_name: str,
):
    """Generate Conv loops (1D, 2D, 3D)."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()

    in_shape = in_tensor.shape
    w_shape = w_tensor.shape
    out_shape = out_tensor.shape

    spatial_dims = len(in_shape) - 2

    group = get_attribute(node, "group", 1)
    strides = get_attribute(node, "strides", [1] * spatial_dims)
    dilations = get_attribute(node, "dilations", [1] * spatial_dims)
    pads = get_attribute(node, "pads", [0] * (spatial_dims * 2))

    batch = in_shape[0]
    in_c = in_shape[1]
    out_c = w_shape[0]

    print(f"node.attributes={node.attributes}")
    print(f"group={group} in_c={in_c} out_c={out_c}")
    is_depthwise = (
        int(group if type(group) is int else group.value if hasattr(group, "value") else group)
        == int(in_c)
        and int(in_c) == int(out_c)
        and int(group if type(group) is int else group.value if hasattr(group, "value") else group)
        > 1
    )
    is_1x1 = (
        spatial_dims == 2
        and w_shape[2] == 1
        and w_shape[3] == 1
        and all(p == 0 for p in pads)
        and all(s == 1 for s in strides)
        and all(d == 1 for d in dilations)
    )

    if is_1x1 and group == 1:
        b.emit("/* 1x1 Conv optimized as MatMul */")
        M = out_c
        K = in_c
        N = 1
        for d in in_shape[2:]:
            N *= d

        b.emit("int m, n, k, b_idx;")
        b.emit(f"for (b_idx = 0; b_idx < {batch}; ++b_idx) {{")
        b.push_indent()
        b.emit(f"for (m = 0; m < {M}; ++m) {{")
        b.push_indent()
        b.emit(f"for (n = 0; n < {N}; ++n) {{")
        b.push_indent()
        b.emit("float sum = 0.0f;")
        b.emit(f"for (k = 0; k < {K}; ++k) {{")
        b.push_indent()
        b.emit(f"sum += {in_name}[b_idx * {K * N} + k * {N} + n] * {w_name}[m * {K} + k];")
        b.pop_indent()
        b.emit("}")

        b.emit(f"{out_name}[b_idx * {M * N} + m * {N} + n] = sum;")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
    elif spatial_dims == 2:
        if is_depthwise:
            b.emit("/* Depthwise Conv2D */")
        else:
            b.emit("/* Naive Conv2D (7-level nested loop) */")

        IH, IW = in_shape[2], in_shape[3]
        KH, KW = w_shape[2], w_shape[3]
        OH, OW = out_shape[2], out_shape[3]
        SH, SW = strides[0], strides[1]
        DH, DW = dilations[0], dilations[1]
        PT, PL = pads[0], pads[1]

        b.emit("int b_idx, oc, ic, oh, ow, kh, kw;")
        b.emit(f"for (b_idx = 0; b_idx < {batch}; ++b_idx) {{")
        b.push_indent()
        b.emit(f"for (oc = 0; oc < {out_c}; ++oc) {{")
        b.push_indent()
        b.emit(f"for (oh = 0; oh < {OH}; ++oh) {{")
        b.push_indent()
        b.emit(f"for (ow = 0; ow < {OW}; ++ow) {{")
        b.push_indent()

        b.emit("float sum = 0.0f;")

        if is_depthwise:
            b.emit(f"for (kh = 0; kh < {KH}; ++kh) {{")
            b.push_indent()
            b.emit(f"for (kw = 0; kw < {KW}; ++kw) {{")
            b.push_indent()
            b.emit(f"int ih = oh * {SH} + kh * {DH} - {PT};")
            b.emit(f"int iw = ow * {SW} + kw * {DW} - {PL};")
            b.emit(f"if (ih >= 0 && ih < {IH} && iw >= 0 && iw < {IW}) {{")
            b.push_indent()
            b.emit(
                f"sum += {in_name}[b_idx * {in_c * IH * IW} + oc * {IH * IW} + ih * {IW} + iw] * {w_name}[oc * {KH * KW} + kh * {KW} + kw];"
            )
            b.pop_indent()
            b.emit("}")
            b.pop_indent()
            b.emit("}")
            b.pop_indent()
            b.emit("}")
        else:
            in_c_per_group = in_c // group
            out_c_per_group = out_c // group
            b.emit(f"int g = oc / {out_c_per_group};")
            b.emit(f"for (ic = 0; ic < {in_c_per_group}; ++ic) {{")
            b.push_indent()
            b.emit(f"for (kh = 0; kh < {KH}; ++kh) {{")
            b.push_indent()
            b.emit(f"for (kw = 0; kw < {KW}; ++kw) {{")
            b.push_indent()
            b.emit(f"int ih = oh * {SH} + kh * {DH} - {PT};")
            b.emit(f"int iw = ow * {SW} + kw * {DW} - {PL};")
            b.emit(f"if (ih >= 0 && ih < {IH} && iw >= 0 && iw < {IW}) {{")
            b.push_indent()
            b.emit(
                f"sum += {in_name}[b_idx * {in_c * IH * IW} + (g * {in_c_per_group} + ic) * {IH * IW} + ih * {IW} + iw] * {w_name}[oc * {in_c_per_group * KH * KW} + ic * {KH * KW} + kh * {KW} + kw];"
            )
            b.pop_indent()
            b.emit("}")
            b.pop_indent()
            b.emit("}")
            b.pop_indent()
            b.emit("}")
            b.pop_indent()
            b.emit("}")

        b.emit(f"{out_name}[b_idx * {out_c * OH * OW} + oc * {OH * OW} + oh * {OW} + ow] = sum;")

        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")

    elif spatial_dims == 1:
        b.emit("/* Conv1D */")
        IW = in_shape[2]
        KW = w_shape[2]
        OW = out_shape[2]
        SW = strides[0]
        DW = dilations[0]
        PL = pads[0]

        b.emit("int b_idx, oc, ic, ow, kw;")
        b.emit(f"for (b_idx = 0; b_idx < {batch}; ++b_idx) {{")
        b.push_indent()
        b.emit(f"for (oc = 0; oc < {out_c}; ++oc) {{")
        b.push_indent()
        b.emit(f"for (ow = 0; ow < {OW}; ++ow) {{")
        b.push_indent()
        b.emit("float sum = 0.0f;")

        in_c_per_group = in_c // group
        out_c_per_group = out_c // group
        b.emit(f"int g = oc / {out_c_per_group};")
        b.emit(f"for (ic = 0; ic < {in_c_per_group}; ++ic) {{")
        b.push_indent()
        b.emit(f"for (kw = 0; kw < {KW}; ++kw) {{")
        b.push_indent()
        b.emit(f"int iw = ow * {SW} + kw * {DW} - {PL};")
        b.emit(f"if (iw >= 0 && iw < {IW}) {{")
        b.push_indent()
        b.emit(
            f"sum += {in_name}[b_idx * {in_c * IW} + (g * {in_c_per_group} + ic) * {IW} + iw] * {w_name}[oc * {in_c_per_group * KW} + ic * {KW} + kw];"
        )
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")

        b.emit(f"{out_name}[b_idx * {out_c * OW} + oc * {OW} + ow] = sum;")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")

    elif spatial_dims == 3:
        b.emit("/* Conv3D */")
        ID, IH, IW = in_shape[2], in_shape[3], in_shape[4]
        KD, KH, KW = w_shape[2], w_shape[3], w_shape[4]
        OD, OH, OW = out_shape[2], out_shape[3], out_shape[4]
        SD, SH, SW = strides[0], strides[1], strides[2]
        DD, DH, DW = dilations[0], dilations[1], dilations[2]
        P_fT, PT, PL = pads[0], pads[1], pads[2]

        b.emit("int b_idx, oc, ic, od, oh, ow, kd, kh, kw;")
        b.emit(f"for (b_idx = 0; b_idx < {batch}; ++b_idx) {{")
        b.push_indent()
        b.emit(f"for (oc = 0; oc < {out_c}; ++oc) {{")
        b.push_indent()
        b.emit(f"for (od = 0; od < {OD}; ++od) {{")
        b.push_indent()
        b.emit(f"for (oh = 0; oh < {OH}; ++oh) {{")
        b.push_indent()
        b.emit(f"for (ow = 0; ow < {OW}; ++ow) {{")
        b.push_indent()
        b.emit("float sum = 0.0f;")

        in_c_per_group = in_c // group
        out_c_per_group = out_c // group
        b.emit(f"int g = oc / {out_c_per_group};")
        b.emit(f"for (ic = 0; ic < {in_c_per_group}; ++ic) {{")
        b.push_indent()
        b.emit(f"for (kd = 0; kd < {KD}; ++kd) {{")
        b.push_indent()
        b.emit(f"for (kh = 0; kh < {KH}; ++kh) {{")
        b.push_indent()
        b.emit(f"for (kw = 0; kw < {KW}; ++kw) {{")
        b.push_indent()
        b.emit(f"int id = od * {SD} + kd * {DD} - {P_fT};")
        b.emit(f"int ih = oh * {SH} + kh * {DH} - {PT};")
        b.emit(f"int iw = ow * {SW} + kw * {DW} - {PL};")
        b.emit(f"if (id >= 0 && id < {ID} && ih >= 0 && ih < {IH} && iw >= 0 && iw < {IW}) {{")
        b.push_indent()
        b.emit(
            f"sum += {in_name}[b_idx * {in_c * ID * IH * IW} + (g * {in_c_per_group} + ic) * {ID * IH * IW} + id * {IH * IW} + ih * {IW} + iw] * {w_name}[oc * {in_c_per_group * KD * KH * KW} + ic * {KD * KH * KW} + kd * {KH * KW} + kh * {KW} + kw];"
        )
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")

        b.emit(
            f"{out_name}[b_idx * {out_c * OD * OH * OW} + oc * {OD * OH * OW} + od * {OH * OW} + oh * {OW} + ow] = sum;"
        )
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")

    b.pop_indent()
    b.emit("}")


def generate_conv_transpose(
    b: C89Builder,
    node: Node,
    out_tensor: Tensor,
    in_tensor: Tensor,
    w_tensor: Tensor,
    bias_tensor: Tensor,
    in_name: str,
    w_name: str,
    bias_name: str,
    out_name: str,
):
    """Generate ConvTranspose loops."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()

    in_shape = in_tensor.shape
    w_shape = w_tensor.shape
    out_shape = out_tensor.shape

    spatial_dims = len(in_shape) - 2

    group = get_attribute(node, "group", 1)
    strides = get_attribute(node, "strides", [1] * spatial_dims)
    dilations = get_attribute(node, "dilations", [1] * spatial_dims)
    pads = get_attribute(node, "pads", [0] * (spatial_dims * 2))

    batch = in_shape[0]
    in_c = in_shape[1]
    out_c = out_shape[1]

    if spatial_dims == 2:
        b.emit("/* ConvTranspose2D */")
        IH, IW = in_shape[2], in_shape[3]
        KH, KW = w_shape[2], w_shape[3]
        OH, OW = out_shape[2], out_shape[3]
        SH, SW = strides[0], strides[1]
        DH, DW = dilations[0], dilations[1]
        PT, PL = pads[0], pads[1]

        b.emit("int b_idx, oc, ic, ih, iw, kh, kw;")
        # Initialize output to zero (since we will accumulate)
        b.emit("int out_size = " + str(batch * out_c * OH * OW) + ";")
        b.emit("int i;")
        b.emit("for (i = 0; i < out_size; ++i) {")
        b.push_indent()
        b.emit(f"{out_name}[i] = 0.0f;")
        b.pop_indent()
        b.emit("}")

        b.emit(f"for (b_idx = 0; b_idx < {batch}; ++b_idx) {{")
        b.push_indent()
        b.emit(f"for (ic = 0; ic < {in_c}; ++ic) {{")
        b.push_indent()
        b.emit(f"for (ih = 0; ih < {IH}; ++ih) {{")
        b.push_indent()
        b.emit(f"for (iw = 0; iw < {IW}; ++iw) {{")
        b.push_indent()

        in_c_per_group = in_c // group
        out_c_per_group = out_c // group
        b.emit(f"int g = ic / {in_c_per_group};")
        b.emit(f"for (oc = 0; oc < {out_c_per_group}; ++oc) {{")
        b.push_indent()
        b.emit(f"for (kh = 0; kh < {KH}; ++kh) {{")
        b.push_indent()
        b.emit(f"for (kw = 0; kw < {KW}; ++kw) {{")
        b.push_indent()

        b.emit(f"int oh = ih * {SH} + kh * {DH} - {PT};")
        b.emit(f"int ow = iw * {SW} + kw * {DW} - {PL};")

        b.emit(f"if (oh >= 0 && oh < {OH} && ow >= 0 && ow < {OW}) {{")
        b.push_indent()
        b.emit(
            f"float in_val = {in_name}[b_idx * {in_c * IH * IW} + ic * {IH * IW} + ih * {IW} + iw];"
        )
        b.emit(
            f"float w_val = {w_name}[ic * {out_c_per_group * KH * KW} + oc * {KH * KW} + kh * {KW} + kw];"
        )
        b.emit(
            f"{out_name}[b_idx * {out_c * OH * OW} + (g * {out_c_per_group} + oc) * {OH * OW} + oh * {OW} + ow] += in_val * w_val;"
        )
        b.pop_indent()
        b.emit("}")

        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")

        if bias_name:
            b.emit("/* Add Bias */")
            b.emit(f"for (b_idx = 0; b_idx < {batch}; ++b_idx) {{")
            b.push_indent()
            b.emit(f"for (oc = 0; oc < {out_c}; ++oc) {{")
            b.push_indent()
            b.emit(f"for (ih = 0; ih < {OH}; ++ih) {{")
            b.push_indent()
            b.emit(f"for (iw = 0; iw < {OW}; ++iw) {{")
            b.push_indent()
            b.emit(
                f"{out_name}[b_idx * {out_c * OH * OW} + oc * {OH * OW} + ih * {OW} + iw] += {bias_name}[oc];"
            )
            b.pop_indent()
            b.emit("}")
            b.pop_indent()
            b.emit("}")
            b.pop_indent()
            b.emit("}")
            b.pop_indent()
            b.emit("}")

    else:
        b.emit("/* Unsupported ConvTranspose dim */")

    b.pop_indent()
    b.emit("}")
