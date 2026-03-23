"""Operation implementations for ONNX to C89 generation."""

import struct

from onnx9000.c_compiler.ast_builder import C89Builder
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Node, Tensor
from onnx9000.core.profiler import resolve_volume


def get_strides(shape):
    if not shape:
        return []
    strides = []
    s = 1
    for dim in reversed(shape):
        strides.insert(0, s)
        if isinstance(dim, int):
            s *= dim
        else:
            s = f"({s} * {dim})"
    return strides


def resolve_broadcast_indices(out_shape, in_shape, loop_var="i"):
    if not in_shape or resolve_volume(in_shape) == 1:
        return "0"
    if in_shape == out_shape:
        return loop_var

    out_strides = get_strides(out_shape)
    in_strides = get_strides(in_shape)

    pad_len = len(out_shape) - len(in_shape)
    padded_in_shape = [1] * pad_len + list(in_shape)
    padded_in_strides = [0] * pad_len + in_strides

    parts = []
    for d in range(len(out_shape)):
        if padded_in_shape[d] != 1:
            out_stride = out_strides[d]
            out_dim = out_shape[d]
            in_stride = padded_in_strides[d]

            coord = (
                f"({loop_var} % {out_dim})"
                if out_stride == 1
                else f"(({loop_var} / {out_stride}) % {out_dim})"
            )

            if in_stride == 1:
                parts.append(coord)
            if in_stride == 1:
                parts.append(coord)

    return " + ".join(parts)


def generate_elementwise_binary(
    b: C89Builder,
    node: Node,
    op_char: str,
    out_tensor: Tensor,
    in1_tensor: Tensor,
    in2_tensor: Tensor,
    in1: str,
    in2: str,
    out: str,
):
    b.emit(f"/* Elementwise {node.op_type} */")
    b.emit("{")
    b.push_indent()
    i_var = b.new_var("i")
    b.emit(f"int {i_var};")
    size_var = str(resolve_volume(out_tensor.shape)) if out_tensor and out_tensor.shape else "1"

    is_int_div_pow2 = False
    pow2_shift = 0
    if (
        op_char == "/"
        and in2_tensor
        and (
            getattr(in2_tensor, "is_initializer", False) or type(in2_tensor).__name__ == "Constant"
        )
        and getattr(in2_tensor, "data", None)
    ):
        if in2_tensor.dtype in (DType.INT32, DType.INT64):
            if resolve_volume(in2_tensor.shape) == 1:
                fmt = "i" if in2_tensor.dtype == DType.INT32 else "q"
                val = struct.unpack(f"<{fmt}", in2_tensor.data[: struct.calcsize(f"<{fmt}")])[0]
                if val > 0 and (val & (val - 1)) == 0:
                    is_int_div_pow2 = True
                    pow2_shift = val.bit_length() - 1

    b.emit(f"for ({i_var} = 0; {i_var} < {size_var}; ++{i_var}) {{")
    b.push_indent()

    idx1 = resolve_broadcast_indices(
        out_tensor.shape, in1_tensor.shape if in1_tensor else [], i_var
    )
    idx2 = resolve_broadcast_indices(
        out_tensor.shape, in2_tensor.shape if in2_tensor else [], i_var
    )

    if is_int_div_pow2:
        b.emit(f"{out}[{i_var}] = {in1}[{idx1}] >> {pow2_shift}; /* Optimized Div */")
    else:
        if op_char == "/":
            b.emit(
                f"{out}[{i_var}] = {in1}[{idx1}] / ({in2}[{idx2}] + 1e-7f); /* Protect DivByZero */"
            )
        else:
            b.emit(f"{out}[{i_var}] = {in1}[{idx1}] {op_char} {in2}[{idx2}];")

    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_math_call(
    b: C89Builder,
    node: Node,
    func_name: str,
    out_tensor: Tensor,
    in1_tensor: Tensor,
    in1: str,
    out: str,
    fallback_macro: bool = False,
):
    b.emit(f"/* Math {node.op_type} */")
    b.emit("{")
    b.push_indent()
    i_var = b.new_var("i")
    b.emit(f"int {i_var};")
    size_var = str(resolve_volume(out_tensor.shape)) if out_tensor and out_tensor.shape else "1"

    b.emit(f"for ({i_var} = 0; {i_var} < {size_var}; ++{i_var}) {{")
    b.push_indent()
    idx1 = resolve_broadcast_indices(
        out_tensor.shape, in1_tensor.shape if in1_tensor else [], i_var
    )
    if fallback_macro:
        b.emit(f"{out}[{i_var}] = ONNX9000_FALLBACK_{func_name.upper()}({in1}[{idx1}]);")
    else:
        b.emit(f"{out}[{i_var}] = {func_name}({in1}[{idx1}]);")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_math_binary_call(
    b: C89Builder,
    node: Node,
    func_name: str,
    out_tensor: Tensor,
    in1_tensor: Tensor,
    in2_tensor: Tensor,
    in1: str,
    in2: str,
    out: str,
):
    b.emit(f"/* Math Binary {node.op_type} */")
    b.emit("{")
    b.push_indent()
    i_var = b.new_var("i")
    b.emit(f"int {i_var};")
    size_var = str(resolve_volume(out_tensor.shape)) if out_tensor and out_tensor.shape else "1"

    idx1 = resolve_broadcast_indices(
        out_tensor.shape, in1_tensor.shape if in1_tensor else [], i_var
    )
    idx2 = resolve_broadcast_indices(
        out_tensor.shape, in2_tensor.shape if in2_tensor else [], i_var
    )

    b.emit(f"for ({i_var} = 0; {i_var} < {size_var}; ++{i_var}) {{")
    b.push_indent()
    b.emit(f"{out}[{i_var}] = {func_name}({in1}[{idx1}], {in2}[{idx2}]);")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_math_unary_op(
    b: C89Builder,
    node: Node,
    op_char: str,
    out_tensor: Tensor,
    in1_tensor: Tensor,
    in1: str,
    out: str,
):
    b.emit(f"/* Unary {node.op_type} */")
    b.emit("{")
    b.push_indent()
    i_var = b.new_var("i")
    b.emit(f"int {i_var};")
    size_var = str(resolve_volume(out_tensor.shape)) if out_tensor and out_tensor.shape else "1"

    b.emit(f"for ({i_var} = 0; {i_var} < {size_var}; ++{i_var}) {{")
    b.push_indent()
    idx1 = resolve_broadcast_indices(
        out_tensor.shape, in1_tensor.shape if in1_tensor else [], i_var
    )
    b.emit(f"{out}[{i_var}] = {op_char}({in1}[{idx1}]);")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_sign(
    b: C89Builder, node: Node, out_tensor: Tensor, in1_tensor: Tensor, in1: str, out: str
):
    b.emit(f"/* Sign {node.op_type} */")
    b.emit("{")
    b.push_indent()
    i_var = b.new_var("i")
    b.emit(f"int {i_var};")
    size_var = str(resolve_volume(out_tensor.shape)) if out_tensor and out_tensor.shape else "1"

    b.emit(f"for ({i_var} = 0; {i_var} < {size_var}; ++{i_var}) {{")
    b.push_indent()
    idx1 = resolve_broadcast_indices(
        out_tensor.shape, in1_tensor.shape if in1_tensor else [], i_var
    )
    b.emit(f"{out}[{i_var}] = ({in1}[{idx1}] > 0) - ({in1}[{idx1}] < 0);")
    b.pop_indent()
    b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_matmul(
    b: C89Builder,
    node: Node,
    out_tensor: Tensor,
    in1_tensor: Tensor,
    in2_tensor: Tensor,
    in1: str,
    in2: str,
    out: str,
    transA: bool = False,
    transB: bool = False,
    alpha: float = 1.0,
    beta: float = 0.0,
    bias: str = "",
    is_integer: bool = False,
):
    """Generate MatMul, BatchMatMul, and Gemm operations."""
    b.emit(f"/* {node.op_type} */")
    b.emit("{")
    b.push_indent()

    shapeA = in1_tensor.shape if in1_tensor else [1, 1]
    shapeB = in2_tensor.shape if in2_tensor else [1, 1]

    batch_size = 1
    if len(shapeA) > 2:
        for dim in shapeA[:-2]:
            batch_size *= dim if isinstance(dim, int) else 1

    M = shapeA[-2] if not transA else shapeA[-1]
    K = shapeA[-1] if not transA else shapeA[-2]
    N = shapeB[-1] if not transB else shapeB[-2]

    acc_type = "int32_t" if is_integer else "float"

    is_gemv = (M == 1 or N == 1) and batch_size == 1

    if (
        batch_size == 1
        and isinstance(M, int)
        and isinstance(K, int)
        and isinstance(N, int)
        and M * K * N <= 27
    ):
        b.emit(f"/* Small MatMul Unrolled ({M}x{K}x{N}) */")
        for m in range(M):
            for n in range(N):
                acc_expr = []
                for k in range(K):
                    idxA = k * M + m if transA else m * K + k
                    idxB = n * K + k if transB else k * N + n
                    acc_expr.append(f"({in1}[{idxA}] * {in2}[{idxB}])")

                out_idx = m * N + n
                sum_expr = " + ".join(acc_expr)
                if alpha != 1.0:
                    sum_expr = f"({sum_expr}) * {alpha}f"
                if bias:
                    sum_expr = (
                        f"{sum_expr} + ({bias}[{n}] * {beta}f)"
                        if beta != 1.0
                        else f"{sum_expr} + {bias}[{n}]"
                    )
                b.emit(f"{out}[{out_idx}] = {sum_expr};")
    elif is_gemv and not is_integer:
        b.emit(f"/* GEMV Optimized ({M}x{K}x{N}) */")
        b.emit("int m, k, n;")
        b.emit(f"int M = {M}, K = {K}, N = {N};")
        b.emit("for (m = 0; m < M; ++m) {")
        b.push_indent()
        b.emit("for (n = 0; n < N; ++n) {")
        b.push_indent()
        b.emit(f"{acc_type} sum = 0;")
        b.emit("for (k = 0; k < K; ++k) {")
        b.push_indent()
        idxA = "k * M + m" if transA else "m * K + k"
        idxB = "n * K + k" if transB else "k * N + n"
        b.emit(f"sum += {in1}[{idxA}] * {in2}[{idxB}];")
        b.pop_indent()
        b.emit("}")
        out_idx = "m * N + n"
        sum_expr = "sum"
        if bias:
            sum_expr = (
                f"{sum_expr} + ({bias}[n] * {beta}f)" if beta != 1.0 else f"{sum_expr} + {bias}[n]"
            )
        b.emit(f"{out}[{out_idx}] = {sum_expr};")
        b.pop_indent()
        b.emit("}")
        b.pop_indent()
        b.emit("}")
    b.pop_indent()
    b.emit("}")


def generate_einsum(b, node, out_tensor, in_tensors, out, ins):
    """Compile Einsum cleanly."""
    equation = node.attributes.get("equation", "")
    if isinstance(equation, bytes):
        equation = equation.decode("utf-8")
    elif not isinstance(equation, str):
        equation = ""

    if "->" not in equation:
        # Fallback or invalid
        return

    lhs, rhs = equation.split("->")
    inputs_eq = lhs.split(",")

    dim_map = {}
    for i, eq in enumerate(inputs_eq):
        shape = in_tensors[i].shape
        for j, char in enumerate(eq):
            dim_map[char] = shape[j]

    from onnx9000.core.profiler import resolve_volume

    b.emit(f"/* Einsum: {equation} */")
    b.emit("{")
    b.push_indent()

    size_var = str(resolve_volume(out_tensor.shape))
    b.emit(f"memset({out}, 0, {size_var} * sizeof(float));")

    c_vars = {}
    for char in dim_map:
        c_var = b.new_var(f"e_{char}")
        c_vars[char] = c_var
        b.emit(f"int {c_var};")

    for char, limit in dim_map.items():
        c_var = c_vars[char]
        b.emit(f"for ({c_var} = 0; {c_var} < {limit}; ++{c_var}) {{")
        b.push_indent()

    def get_index_expr(eq, shape):
        exprs = []
        stride = 1
        for i in reversed(range(len(eq))):
            char = eq[i]
            c_var = c_vars[char]
            if stride == 1:
                exprs.append(c_var)
            else:
                exprs.append(f"({c_var} * {stride})")
            stride *= shape[i]
        return " + ".join(exprs) if exprs else "0"

    out_idx = get_index_expr(rhs, out_tensor.shape)

    term = []
    for i, eq in enumerate(inputs_eq):
        in_idx = get_index_expr(eq, in_tensors[i].shape)
        term.append(f"{ins[i]}[{in_idx}]")

    b.emit(f"{out}[{out_idx}] += " + " * ".join(term) + ";")

    for _ in dim_map:
        b.pop_indent()
        b.emit("}")

    b.pop_indent()
    b.emit("}")
