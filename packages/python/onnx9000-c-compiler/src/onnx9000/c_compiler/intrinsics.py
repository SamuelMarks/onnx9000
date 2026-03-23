"""Hardware-Specific Intrinsics & Pragmas generation logic for ONNX to C."""

from onnx9000.c_compiler.ast_builder import C89Builder
from onnx9000.core.ir import Node


def emit_cmsis_nn_qlinear_matmul(
    b: C89Builder,
    node: Node,
    in1: str,
    in2: str,
    out: str,
    M: int,
    K: int,
    N: int,
    zp1: int,
    zp2: int,
    zp_out: int,
    multiplier: float,
):
    """Emit CMSIS-NN specific arm_fully_connected_s8 call."""
    b.emit(f"/* CMSIS-NN arm_fully_connected_s8 optimized ({M}x{K}x{N}) */")
    # For now, this is a conceptual structural mapping based on the CMSIS-NN API
    b.emit("cmsis_nn_context ctx;")
    b.emit("ctx.buf = NULL;")
    b.emit("ctx.size = 0;")

    b.emit("cmsis_nn_fc_params fc_params;")
    b.emit(f"fc_params.input_offset = -{zp1};")
    b.emit(f"fc_params.filter_offset = -{zp2};")
    b.emit(f"fc_params.output_offset = {zp_out};")

    # Calculate quant multiplier and shift offline for the C generation
    b.emit(
        f"/* Offline calculation needed for multiplier {multiplier} into (int32_t mult, int32_t shift) */"
    )
    b.emit("cmsis_nn_quantized_activation quant_params;")
    b.emit("quant_params.multiplier = 1; /* Placeholder */")
    b.emit("quant_params.shift = 0; /* Placeholder */")

    b.emit("cmsis_nn_dims input_dims = {1, 1, 1, " + str(K) + "};")
    b.emit("cmsis_nn_dims filter_dims = {1, 1, " + str(N) + ", " + str(K) + "};")
    b.emit("cmsis_nn_dims bias_dims = {1, 1, 1, " + str(N) + "};")
    b.emit("cmsis_nn_dims output_dims = {1, 1, 1, " + str(N) + "};")

    b.emit(
        f"arm_fully_connected_s8(&ctx, &fc_params, &quant_params, &input_dims, {in1}, &filter_dims, {in2}, &bias_dims, NULL, &output_dims, {out});"
    )


def emit_cmsis_nn_qlinear_conv(
    b: C89Builder,
    node: Node,
    in1: str,
    w: str,
    bias: str,
    out: str,
    IH: int,
    IW: int,
    KH: int,
    KW: int,
    OH: int,
    OW: int,
    in_c: int,
    out_c: int,
    zp1: int,
    zp2: int,
    zp_out: int,
    multiplier: float,
    strides: list,
    pads: list,
    dilations: list,
):
    """Emit CMSIS-NN specific arm_convolve_s8 call."""
    b.emit("/* CMSIS-NN arm_convolve_s8 optimized */")
    b.emit("cmsis_nn_context ctx;")
    b.emit("ctx.buf = NULL;")
    b.emit("ctx.size = 0;")

    b.emit("cmsis_nn_conv_params conv_params;")
    b.emit(f"conv_params.input_offset = -{zp1};")
    b.emit(f"conv_params.output_offset = {zp_out};")
    b.emit(f"conv_params.stride.h = {strides[0]};")
    b.emit(f"conv_params.stride.w = {strides[1]};")
    b.emit(f"conv_params.padding.h = {pads[0]};")
    b.emit(f"conv_params.padding.w = {pads[1]};")
    b.emit(f"conv_params.dilation.h = {dilations[0]};")
    b.emit(f"conv_params.dilation.w = {dilations[1]};")
    b.emit("conv_params.activation.min = -128;")
    b.emit("conv_params.activation.max = 127;")

    b.emit("cmsis_nn_quantized_activation quant_params;")
    b.emit("quant_params.multiplier = 1; /* Placeholder */")
    b.emit("quant_params.shift = 0; /* Placeholder */")

    b.emit("cmsis_nn_dims input_dims = {1, " + str(IH) + ", " + str(IW) + ", " + str(in_c) + "};")
    b.emit(
        "cmsis_nn_dims filter_dims = {"
        + str(out_c)
        + ", "
        + str(KH)
        + ", "
        + str(KW)
        + ", "
        + str(in_c)
        + "};"
    )
    b.emit("cmsis_nn_dims bias_dims = {1, 1, 1, " + str(out_c) + "};")
    b.emit("cmsis_nn_dims output_dims = {1, " + str(OH) + ", " + str(OW) + ", " + str(out_c) + "};")

    bias_str = bias if bias else "NULL"
    b.emit(
        f"arm_convolve_s8(&ctx, &conv_params, &quant_params, &input_dims, {in1}, &filter_dims, {w}, &bias_dims, {bias_str}, &output_dims, {out});"
    )


def apply_simd_unroll(b: C89Builder, target: str):
    """Generate SIMD loop unrolling hints based on target platform."""
    if target == "desktop" or target == "avx2":
        b.emit("#pragma omp parallel for")
        b.emit("#pragma GCC unroll 4")
    elif target == "riscv-v":
        # Specific vector unroll hints for riscv gcc
        b.emit("#pragma GCC unroll 8")


def emit_esp_nn_qlinear_matmul(
    b: C89Builder,
    node: Node,
    in1: str,
    in2: str,
    out: str,
    M: int,
    K: int,
    N: int,
    zp1: int,
    zp2: int,
    zp_out: int,
    multiplier: float,
):
    b.emit(f"/* ESP-NN esp_nn_fully_connected_s8 optimized ({M}x{K}x{N}) */")
    b.emit(f"/* Offline calculation needed for multiplier {multiplier} */")
    b.emit("int32_t out_mult = 1; /* Placeholder */")
    b.emit("int32_t out_shift = 0; /* Placeholder */")
    b.emit(
        f"esp_nn_fully_connected_s8({in1}, -{zp1}, {K}, {in2}, -{zp2}, NULL, {out}, {zp_out}, out_mult, out_shift, -128, 127, {M}, {N});"
    )


def emit_esp_nn_qlinear_conv(
    b: C89Builder,
    node: Node,
    in1: str,
    w: str,
    bias: str,
    out: str,
    IH: int,
    IW: int,
    KH: int,
    KW: int,
    OH: int,
    OW: int,
    in_c: int,
    out_c: int,
    zp1: int,
    zp2: int,
    zp_out: int,
    multiplier: float,
    strides: list,
    pads: list,
    dilations: list,
):
    b.emit("/* ESP-NN esp_nn_conv_s8 optimized */")
    b.emit("int32_t out_mult = 1; /* Placeholder */")
    b.emit("int32_t out_shift = 0; /* Placeholder */")
    bias_str = bias if bias else "NULL"
    b.emit(
        f"esp_nn_conv_s8({in1}, {IH}, {IW}, {in_c}, -{zp1}, {w}, {KH}, {KW}, -{zp2}, {bias_str}, {out}, {OH}, {OW}, {out_c}, {zp_out}, out_mult, out_shift, -128, 127, {strides[0]}, {strides[1]}, {pads[0]}, {pads[1]}, {pads[2]}, {pads[3]});"
    )


def emit_avx2_headers(b: C89Builder, target: str):
    if target == "avx2":
        b.emit("#ifdef __AVX2__")
        b.emit("#include <immintrin.h>")
        b.emit("#endif")
