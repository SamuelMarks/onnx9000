"""Quantization specific logic for ONNX to C89 generation."""

from onnx9000.c_compiler.ast_builder import C89Builder
from onnx9000.c_compiler.operations import resolve_broadcast_indices
from onnx9000.core.ir import Node, Tensor
from onnx9000.core.profiler import resolve_volume


def generate_quantize_linear(
    b: C89Builder,
    node: Node,
    out_tensor: Tensor,
    in_tensor: Tensor,
    scale_tensor: Tensor,
    zp_tensor: Tensor,
    in_name: str,
    scale_name: str,
    zp_name: str,
    out_name: str,
):
    """Generate QuantizeLinear."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    b.emit("int i;")
    size_var = str(resolve_volume(out_tensor.shape)) if out_tensor and out_tensor.shape else "1"

    b.emit(f"for (i = 0; i < {size_var}; ++i) {{")
    b.push_indent()

    idx_scale = resolve_broadcast_indices(
        out_tensor.shape, scale_tensor.shape if scale_tensor else []
    )

    # Cast to float, divide by scale, round, add zero point, then cast to uint8/int8
    b.emit(f"float scaled = {in_name}[i] / {scale_name}[{idx_scale}];")
    # Native C89 round mapping
    b.emit(
        "float rounded = scaled >= 0.0f ? (float)(int)(scaled + 0.5f) : (float)(int)(scaled - 0.5f);"
    )

    if zp_name:
        idx_zp = resolve_broadcast_indices(out_tensor.shape, zp_tensor.shape if zp_tensor else [])
        b.emit(f"float qval = rounded + (float)({zp_name}[{idx_zp}]);")

    b.emit("if (qval > 255.0f) qval = 255.0f;")
    b.emit("if (qval < -128.0f) qval = -128.0f;")

    from onnx9000.core.dtypes import DType

    if out_tensor.dtype == DType.UINT8:
        b.emit("if (qval < 0.0f) qval = 0.0f;")
        b.emit(f"{out_name}[i] = (uint8_t)qval;")
    else:
        b.emit(f"{out_name}[i] = (int8_t)qval;")

    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_dequantize_linear(
    b: C89Builder,
    node: Node,
    out_tensor: Tensor,
    in_tensor: Tensor,
    scale_tensor: Tensor,
    zp_tensor: Tensor,
    in_name: str,
    scale_name: str,
    zp_name: str,
    out_name: str,
):
    """Generate DequantizeLinear."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()
    b.emit("int i;")
    size_var = str(resolve_volume(out_tensor.shape)) if out_tensor and out_tensor.shape else "1"

    b.emit(f"for (i = 0; i < {size_var}; ++i) {{")
    b.push_indent()

    idx_scale = resolve_broadcast_indices(
        out_tensor.shape, scale_tensor.shape if scale_tensor else []
    )

    if zp_name:
        idx_zp = resolve_broadcast_indices(out_tensor.shape, zp_tensor.shape if zp_tensor else [])
        b.emit(
            f"{out_name}[i] = ((float){in_name}[i] - (float){zp_name}[{idx_zp}]) * {scale_name}[{idx_scale}];"
        )

    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_qlinear_matmul(
    b: C89Builder,
    node: Node,
    out_tensor: Tensor,
    in1_tensor: Tensor,
    scale1_tensor: Tensor,
    zp1_tensor: Tensor,
    in2_tensor: Tensor,
    scale2_tensor: Tensor,
    zp2_tensor: Tensor,
    scale_out_tensor: Tensor,
    zp_out_tensor: Tensor,
    in1: str,
    s1: str,
    zp1: str,
    in2: str,
    s2: str,
    zp2: str,
    s_out: str,
    zp_out: str,
    out: str,
):
    """Generate QLinearMatMul loops using precision-safe INT32 accumulators."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()

    shapeA = in1_tensor.shape if in1_tensor else [1, 1]
    shapeB = in2_tensor.shape if in2_tensor else [1, 1]

    batch_size = 1
    if len(shapeA) > 2:
        for dim in shapeA[:-2]:
            batch_size *= dim if isinstance(dim, int) else 1

    M = shapeA[-2]
    K = shapeA[-1]
    N = shapeB[-1]

    b.emit("/* Re-quantization logic */")

    import struct

    s1_val = (
        struct.unpack("<f", scale1_tensor.data[:4])[0]
        if getattr(scale1_tensor, "data", None)
        else 1.0
    )
    s2_val = (
        struct.unpack("<f", scale2_tensor.data[:4])[0]
        if getattr(scale2_tensor, "data", None)
        else 1.0
    )
    s_out_val = (
        struct.unpack("<f", scale_out_tensor.data[:4])[0]
        if getattr(scale_out_tensor, "data", None)
        else 1.0
    )

    real_multiplier = (s1_val * s2_val) / s_out_val
    b.emit(
        f"float real_multiplier = {real_multiplier}f; /* Pre-calculated M = (S1 * S2) / S_out */"
    )

    zp1_val = (
        struct.unpack("<B", zp1_tensor.data[:1])[0] if getattr(zp1_tensor, "data", None) else 0
    )
    zp2_val = (
        struct.unpack("<B", zp2_tensor.data[:1])[0] if getattr(zp2_tensor, "data", None) else 0
    )
    zp_out_val = (
        struct.unpack("<B", zp_out_tensor.data[:1])[0]
        if getattr(zp_out_tensor, "data", None)
        else 0
    )

    b.emit("int b, m, n, k;")
    b.emit(f"int M = {M}, K = {K}, N = {N};")
    b.emit(f"int32_t zp1 = {zp1_val};")
    b.emit(f"int32_t zp2 = {zp2_val};")
    b.emit(f"int32_t zp_out = {zp_out_val};")

    b.emit(f"for (b = 0; b < {batch_size}; ++b) {{")
    b.push_indent()
    b.emit("for (m = 0; m < M; ++m) {")
    b.push_indent()
    b.emit("for (n = 0; n < N; ++n) {")
    b.push_indent()
    from onnx9000.c_compiler.spatial import get_attribute

    use_block_q4_0 = get_attribute(node, "use_block_q4_0", 0)

    if use_block_q4_0:
        b.emit("float float_out = 0.0f;")
        b.emit(f"/* Assuming block_q4_0/block_q8_0 format for {in1} and {in2} */")
        b.emit(
            f"ggml_vec_dot_q4_0_q8_0(K, &float_out, &((const block_q4_0*){in1})[b * M * (K/32) + m * (K/32)], &((const block_q8_0*){in2})[b * N * (K/32) + n * (K/32)]);"
        )
    else:
        b.emit("int32_t sum = 0;")
        b.emit("for (k = 0; k < K; ++k) {")
        b.push_indent()
        idxA = "b * M * K + (m * K + k)"
        idxB = "b * K * N + (k * N + n)"
        b.emit(f"sum += ((int32_t){in1}[{idxA}] - zp1) * ((int32_t){in2}[{idxB}] - zp2);")
        b.pop_indent()
        b.emit("}")
        b.emit("float float_out = (float)sum * real_multiplier;")

    out_idx = "b * M * N + (m * N + n)"
    b.emit(
        "float rounded = float_out >= 0.0f ? (float)(int)(float_out + 0.5f) : (float)(int)(float_out - 0.5f);"
    )
    b.emit("int32_t qval = (int32_t)rounded + zp_out;")
    b.emit("if (qval > 255) qval = 255;")
    b.emit("if (qval < 0) qval = 0;")
    b.emit(f"{out}[{out_idx}] = (uint8_t)qval;")

    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")

    b.pop_indent()
    b.emit("}")


def generate_qlinear_conv(
    b: C89Builder,
    node: Node,
    out_tensor: Tensor,
    in_tensor: Tensor,
    scale1_tensor: Tensor,
    zp1_tensor: Tensor,
    w_tensor: Tensor,
    scale2_tensor: Tensor,
    zp2_tensor: Tensor,
    scale_out_tensor: Tensor,
    zp_out_tensor: Tensor,
    bias_tensor: Tensor,
    in1: str,
    s1: str,
    zp1: str,
    in2: str,
    s2: str,
    zp2: str,
    s_out: str,
    zp_out: str,
    bias: str,
    out: str,
):
    """Generate QLinearConv loops using precision-safe INT32 accumulators."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()

    in_shape = in_tensor.shape
    w_shape = w_tensor.shape
    out_shape = out_tensor.shape

    spatial_dims = len(in_shape) - 2
    from onnx9000.c_compiler.spatial import get_attribute

    group = get_attribute(node, "group", 1)
    strides = get_attribute(node, "strides", [1] * spatial_dims)
    dilations = get_attribute(node, "dilations", [1] * spatial_dims)
    pads = get_attribute(node, "pads", [0] * (spatial_dims * 2))

    batch = in_shape[0]
    in_c = in_shape[1]
    out_c = w_shape[0]

    import struct

    s1_val = (
        struct.unpack("<f", scale1_tensor.data[:4])[0]
        if getattr(scale1_tensor, "data", None)
        else 1.0
    )
    s2_val = (
        struct.unpack("<f", scale2_tensor.data[:4])[0]
        if getattr(scale2_tensor, "data", None)
        else 1.0
    )
    s_out_val = (
        struct.unpack("<f", scale_out_tensor.data[:4])[0]
        if getattr(scale_out_tensor, "data", None)
        else 1.0
    )

    real_multiplier = (s1_val * s2_val) / s_out_val
    b.emit(
        f"float real_multiplier = {real_multiplier}f; /* Pre-calculated M = (S1 * S2) / S_out */"
    )

    zp1_val = (
        struct.unpack("<B", zp1_tensor.data[:1])[0] if getattr(zp1_tensor, "data", None) else 0
    )
    zp2_val = (
        struct.unpack("<B", zp2_tensor.data[:1])[0] if getattr(zp2_tensor, "data", None) else 0
    )
    zp_out_val = (
        struct.unpack("<B", zp_out_tensor.data[:1])[0]
        if getattr(zp_out_tensor, "data", None)
        else 0
    )

    if spatial_dims == 2:
        IH, IW = in_shape[2], in_shape[3]
        KH, KW = w_shape[2], w_shape[3]
        OH, OW = out_shape[2], out_shape[3]
        SH, SW = strides[0], strides[1]
        DH, DW = dilations[0], dilations[1]
        PT, PL = pads[0], pads[1]

        b.emit("int b_idx, oc, ic, oh, ow, kh, kw;")
        b.emit(f"int32_t zp1 = {zp1_val};")
        b.emit(f"int32_t zp2 = {zp2_val};")
        b.emit(f"int32_t zp_out = {zp_out_val};")
        b.emit(f"for (b_idx = 0; b_idx < {batch}; ++b_idx) {{")
        b.push_indent()
        b.emit(f"for (oc = 0; oc < {out_c}; ++oc) {{")
        b.push_indent()
        b.emit(f"for (oh = 0; oh < {OH}; ++oh) {{")
        b.push_indent()
        b.emit(f"for (ow = 0; ow < {OW}; ++ow) {{")
        b.push_indent()

        b.emit("int32_t sum = 0;")

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
            f"int32_t in_val = (int32_t){in1}[b_idx * {in_c * IH * IW} + (g * {in_c_per_group} + ic) * {IH * IW} + ih * {IW} + iw] - zp1;"
        )
        b.emit(
            f"int32_t w_val = (int32_t){in2}[oc * {in_c_per_group * KH * KW} + ic * {KH * KW} + kh * {KW} + kw] - zp2;"
        )
        b.emit("sum += in_val * w_val;")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")

        if bias:
            b.emit(f"sum += {bias}[oc]; /* Bias is usually INT32 in QLinearConv */")

        b.emit("float float_out = (float)sum * real_multiplier;")
        b.emit(
            "float rounded = float_out >= 0.0f ? (float)(int)(float_out + 0.5f) : (float)(int)(float_out - 0.5f);"
        )
        b.emit("int32_t qval = (int32_t)rounded + zp_out;")
        b.emit("if (qval > 255) qval = 255;")
        b.emit("if (qval < 0) qval = 0;")
        b.emit(
            f"{out}[b_idx * {out_c * OH * OW} + oc * {OH * OW} + oh * {OW} + ow] = (uint8_t)qval;"
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
