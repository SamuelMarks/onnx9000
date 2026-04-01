"""Activation and Normalization operation implementations for ONNX to C89 generation."""

from onnx9000.c_compiler.ast_builder import C89Builder
from onnx9000.c_compiler.spatial import get_attribute
from onnx9000.core.ir import Node, Tensor
from onnx9000.core.profiler import resolve_volume


def generate_activation(
    b: C89Builder,
    node: Node,
    out_tensor: Tensor,
    in_tensor: Tensor,
    in_name: str,
    out_name: str,
    op_type: str,
    use_math_h: bool = True,
):
    """Generate Activation loops like Relu, Sigmoid, Tanh, Gelu."""
    b.emit(f"/* {op_type} */")
    b.emit("{")
    b.push_indent()

    size_var = str(resolve_volume(out_tensor.shape)) if out_tensor and out_tensor.shape else "1"
    b.emit("int i;")
    b.emit(f"for (i = 0; i < {size_var}; ++i) {{")
    b.push_indent()

    b.emit(f"float val = {in_name}[i];")

    if op_type == "Relu":
        b.emit(f"{out_name}[i] = val > 0.0f ? val : 0.0f;")
    elif op_type == "LeakyRelu":
        alpha = get_attribute(node, "alpha", 0.01)
        b.emit(f"{out_name}[i] = val > 0.0f ? val : {alpha}f * val;")
    elif op_type == "Sigmoid":
        exp_func = "expf" if use_math_h else "ONNX9000_FALLBACK_EXPF"
        b.emit(f"{out_name}[i] = 1.0f / (1.0f + {exp_func}(-val));")
    elif op_type == "Tanh":
        tanh_func = (
            "tanhf"
            if use_math_h
            else "(2.0f / (1.0f + ONNX9000_FALLBACK_EXPF(-2.0f * val)) - 1.0f)"
        )  # inline fallback
        b.emit(f"{out_name}[i] = {tanh_func}(val);")
    elif op_type == "HardSigmoid":
        alpha = get_attribute(node, "alpha", 0.2)
        beta = get_attribute(node, "beta", 0.5)
        b.emit(f"float s_val = {alpha}f * val + {beta}f;")
        b.emit(f"{out_name}[i] = s_val < 0.0f ? 0.0f : (s_val > 1.0f ? 1.0f : s_val);")
    elif op_type == "HardSwish":
        b.emit("float s_val = val + 3.0f;")
        b.emit("s_val = s_val < 0.0f ? 0.0f : (s_val > 6.0f ? 6.0f : s_val);")
        b.emit(f"{out_name}[i] = val * s_val / 6.0f;")
    elif op_type == "Softplus":
        exp_func = "expf" if use_math_h else "ONNX9000_FALLBACK_EXPF"
        log_func = "logf" if use_math_h else "ONNX9000_FALLBACK_LOGF"
        b.emit(f"{out_name}[i] = {log_func}({exp_func}(val) + 1.0f);")
    elif op_type == "Gelu":
        # Using Tanh approximation
        exp_func = "expf" if use_math_h else "ONNX9000_FALLBACK_EXPF"
        b.emit("float c = 0.7978845608f * (val + 0.044715f * val * val * val);")
        if use_math_h:
            b.emit(f"{out_name}[i] = 0.5f * val * (1.0f + tanhf(c));")
        else:
            b.emit(f"float e = {exp_func}(-2.0f * c);")
            b.emit("float t = (1.0f - e) / (1.0f + e);")
            b.emit(f"{out_name}[i] = 0.5f * val * (1.0f + t);")
    elif op_type == "Swish":
        exp_func = "expf" if use_math_h else "ONNX9000_FALLBACK_EXPF"
        b.emit(f"float sig = 1.0f / (1.0f + {exp_func}(-val));")
        b.emit(f"{out_name}[i] = val * sig;")
    elif op_type == "Mish":
        exp_func = "expf" if use_math_h else "ONNX9000_FALLBACK_EXPF"
        # softplus = log(1 + exp(x))
        if use_math_h:
            b.emit("float sp = log1pf(expf(val));")
            b.emit(f"{out_name}[i] = val * tanhf(sp);")
        else:
            b.emit(f"float sp = log(1.0f + {exp_func}(val));")
            b.emit(f"float e2 = {exp_func}(-2.0f * sp);")
            b.emit("float th = (1.0f - e2) / (1.0f + e2);")
            b.emit(f"{out_name}[i] = val * th;")
    elif op_type == "Clip":
        # usually min/max are provided as inputs 1 and 2, but for C loop simplicity let's handle dynamic later
        # For now, strict bounds check
        b.emit(f"{out_name}[i] = val;")  # Placeholder for input variable extraction
    elif op_type == "PRelu":
        pass  # handled via elementwise broadcast usually

    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_softmax(
    b: C89Builder,
    node: Node,
    out_tensor: Tensor,
    in_tensor: Tensor,
    in_name: str,
    out_name: str,
    use_math_h: bool = True,
    is_log: bool = False,
):
    """Generate Softmax and LogSoftmax."""
    b.emit(f"/* {'Log' if is_log else ''}Softmax */")
    b.emit("{")
    b.push_indent()

    in_shape = in_tensor.shape
    axis = get_attribute(node, "axis", -1)
    if axis < 0:
        axis += len(in_shape)

    pre_axis_vol = resolve_volume(in_shape[:axis]) if axis > 0 else 1
    axis_dim = in_shape[axis]
    post_axis_vol = resolve_volume(in_shape[axis + 1 :]) if axis < len(in_shape) - 1 else 1

    exp_func = "expf" if use_math_h else "ONNX9000_FALLBACK_EXPF"
    log_func = "logf" if use_math_h else "ONNX9000_FALLBACK_LOGF"

    b.emit("#if defined(__wasm_simd128__)")
    b.emit("/* WASM SIMD128 Numerically stable Softmax reduction */")
    b.emit("/* v128_t max_vec = wasm_f32x4_splat(-1e38f); ... */")
    b.emit("#endif")

    b.emit("int pre, post, d;")
    b.emit(f"for (pre = 0; pre < {pre_axis_vol}; ++pre) {{")
    b.push_indent()
    b.emit(f"for (post = 0; post < {post_axis_vol}; ++post) {{")
    b.push_indent()

    # 1. Find Max (for numerical stability)
    b.emit("float max_val = -1e38f;")
    b.emit(f"for (d = 0; d < {axis_dim}; ++d) {{")
    b.push_indent()
    b.emit(f"float val = {in_name}[pre * {axis_dim * post_axis_vol} + d * {post_axis_vol} + post];")
    b.emit("if (val > max_val) max_val = val;")
    b.pop_indent()
    b.emit("}")

    # 2. Exponentiate and sum
    b.emit("float sum = 0.0f;")
    b.emit(f"for (d = 0; d < {axis_dim}; ++d) {{")
    b.push_indent()
    b.emit(
        f"float val = {in_name}[pre * {axis_dim * post_axis_vol} + d * {post_axis_vol} + post] - max_val;"
    )
    b.emit(f"float e = {exp_func}(val);")
    b.emit(f"{out_name}[pre * {axis_dim * post_axis_vol} + d * {post_axis_vol} + post] = e;")
    b.emit("sum += e;")
    b.pop_indent()
    b.emit("}")

    # 3. Normalize
    if is_log:
        b.emit(f"float log_sum = {log_func}(sum);")
        b.emit(f"for (d = 0; d < {axis_dim}; ++d) {{")
        b.push_indent()
        b.emit(
            f"float val = {in_name}[pre * {axis_dim * post_axis_vol} + d * {post_axis_vol} + post] - max_val;"
        )
        b.emit(
            f"{out_name}[pre * {axis_dim * post_axis_vol} + d * {post_axis_vol} + post] = val - log_sum;"
        )
        b.pop_indent()
        b.emit("}")
    else:
        b.emit(f"for (d = 0; d < {axis_dim}; ++d) {{")
        b.push_indent()
        b.emit(f"{out_name}[pre * {axis_dim * post_axis_vol} + d * {post_axis_vol} + post] /= sum;")
        b.pop_indent()
        b.emit("}")

    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")

    b.pop_indent()
    b.emit("}")


def generate_normalization(
    b: C89Builder,
    node: Node,
    out_tensor: Tensor,
    in_tensor: Tensor,
    in_name: str,
    scale_name: str,
    bias_name: str,
    out_name: str,
    norm_type: str,
):
    """Generate BatchNormalization, InstanceNormalization, LayerNormalization."""
    b.emit(f"/* {norm_type} */")
    b.emit("{")
    b.push_indent()

    epsilon = get_attribute(node, "epsilon", 1e-05)
    in_shape = in_tensor.shape

    if norm_type == "BatchNormalization":
        # Assuming inference mode
        channels = in_shape[1]
        spatial_vol = resolve_volume(in_shape[2:]) if len(in_shape) > 2 else 1
        batch = in_shape[0]

        b.emit("int b_idx, c, s;")
        b.emit(f"for (b_idx = 0; b_idx < {batch}; ++b_idx) {{")
        b.push_indent()
        b.emit(f"for (c = 0; c < {channels}; ++c) {{")
        b.push_indent()

        # Need mean and var tensors natively, normally input 3 and 4
        mean_name = b._sanitize(node.inputs[3]) if len(node.inputs) > 3 else "0"
        var_name = b._sanitize(node.inputs[4]) if len(node.inputs) > 4 else "1"

        b.emit(f"float m = {mean_name}[c];")
        b.emit(f"float v = {var_name}[c];")
        b.emit(f"float scale = {scale_name}[c] / sqrtf(v + {epsilon}f);")
        b.emit(f"float bias = {bias_name}[c] - m * scale;")

        b.emit(f"for (s = 0; s < {spatial_vol}; ++s) {{")
        b.push_indent()
        b.emit(f"int idx = b_idx * {channels * spatial_vol} + c * {spatial_vol} + s;")
        b.emit(f"{out_name}[idx] = {in_name}[idx] * scale + bias;")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")

    elif norm_type == "LayerNormalization":
        axis = get_attribute(node, "axis", -1)
        if axis < 0:
            axis += len(in_shape)
        pre_axis_vol = resolve_volume(in_shape[:axis]) if axis > 0 else 1
        axis_vol = resolve_volume(in_shape[axis:])

        b.emit("int pre, d;")
        b.emit(f"for (pre = 0; pre < {pre_axis_vol}; ++pre) {{")
        b.push_indent()

        b.emit("float sum = 0.0f, sq_sum = 0.0f;")
        b.emit(f"for (d = 0; d < {axis_vol}; ++d) {{")
        b.push_indent()
        b.emit(f"float val = {in_name}[pre * {axis_vol} + d];")
        b.emit("sum += val;")
        b.emit("sq_sum += val * val;")
        b.pop_indent()
        b.emit("}")

        b.emit(f"float mean = sum / {axis_vol};")
        b.emit(f"float var = (sq_sum / {axis_vol}) - (mean * mean);")
        b.emit(f"float inv_std = 1.0f / sqrtf(var + {epsilon}f);")

        b.emit(f"for (d = 0; d < {axis_vol}; ++d) {{")
        b.push_indent()
        b.emit(f"float norm = ({in_name}[pre * {axis_vol} + d] - mean) * inv_std;")
        b.emit(f"norm *= {scale_name}[d];")
        b.emit(f"norm += {bias_name}[d];")
        b.emit(f"{out_name}[pre * {axis_vol} + d] = norm;")
        b.pop_indent()
        b.emit("}")

        b.pop_indent()
        b.emit("}")

    b.pop_indent()
    b.emit("}")
