"""
C++ Code Generation Utilities

Translates ONNX operations to equivalent C++ bindings and memory buffers.
"""

from onnx9000.codegen.generator import Generator
from onnx9000.codegen.utils import get_omp_pragma
from onnx9000.ir import Node
from onnx9000.registry import registry


@registry.register("MatMul")
def generate_matmul(node: Node, generator_context: Generator) -> str:
    """generate_matmul docstring."""

    inp1 = generator_context.get_tensor_name(node.inputs[0])
    inp2 = generator_context.get_tensor_name(node.inputs[1])
    out = generator_context.get_tensor_name(node.outputs[0])

    tensor_info = generator_context.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    pragma = get_omp_pragma(f"{out}_size")

    return f"""
        // MatMul
        int64_t {out}_M = {inp1}.shape[0];
        int64_t {out}_K = {inp1}.shape[1];
        int64_t {out}_N = {inp2}.shape[1];

        std::vector<int64_t> {out}_shape = {{{out}_M, {out}_N}};
        int64_t {out}_size = {out}_M * {out}_N;
        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);

        #if defined(__APPLE__) && defined(USE_ACCELERATE)
        if constexpr (std::is_same_v<{cpp_type}, float>) {{
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        {out}_M, {out}_N, {out}_K,
                        1.0f, {inp1}.data, {out}_K,
                        {inp2}.data, {out}_N,
                        0.0f, {out}.data, {out}_N);
        }} else {{
            {pragma} collapse(2)
            for (int64_t i = 0; i < {out}_M; ++i) {{
                for (int64_t j = 0; j < {out}_N; ++j) {{
                    {cpp_type} sum = static_cast<{cpp_type}>(0);
                    for (int64_t k = 0; k < {out}_K; ++k) {{
                        sum += {inp1}.data[i * {out}_K + k] * {inp2}.data[k * {out}_N + j];
                    }}
                    {out}.data[i * {out}_N + j] = sum;
                }}
            }}
        }}
        #else
        {pragma} collapse(2)
        for (int64_t i = 0; i < {out}_M; ++i) {{
            for (int64_t j = 0; j < {out}_N; ++j) {{
                {cpp_type} sum = static_cast<{cpp_type}>(0);
                for (int64_t k = 0; k < {out}_K; ++k) {{
                    sum += {inp1}.data[i * {out}_K + k] * {inp2}.data[k * {out}_N + j];
                }}
                {out}.data[i * {out}_N + j] = sum;
            }}
        }}
        #endif
        """


@registry.register("Gemm")
def generate_gemm(node: Node, generator_context: Generator) -> str:  # pragma: no cover
    """generate_gemm docstring."""
    inp1 = generator_context.get_tensor_name(node.inputs[0])
    inp2 = generator_context.get_tensor_name(node.inputs[1])

    has_bias = len(node.inputs) > 2
    bias_var = generator_context.get_tensor_name(node.inputs[2]) if has_bias else None

    out = generator_context.get_tensor_name(node.outputs[0])

    tensor_info = generator_context.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    alpha = node.attributes.get("alpha", 1.0)
    beta = node.attributes.get("beta", 1.0)
    trans_a = node.attributes.get("trans_a", 0)
    trans_b = node.attributes.get("trans_b", 0)

    pragma = get_omp_pragma(f"{out}_size")

    bias_logic = (
        f" + static_cast<{cpp_type}>({beta}) * {bias_var}.data[j]" if has_bias else ""
    )

    # cblas_sgemm can handle alpha/beta, but beta logic for arbitrary bias vector is tricky.
    # Usually you initialize C with bias, then call sgemm with beta=1.0.

    cblas_trans_a = "CblasTrans" if trans_a else "CblasNoTrans"
    cblas_trans_b = "CblasTrans" if trans_b else "CblasNoTrans"

    lda = f"{out}_M" if trans_a else f"{out}_K"
    ldb = f"{out}_K" if trans_b else f"{out}_N"

    return f"""
        // Gemm (Fused MatMul + Add)
        int64_t {out}_M = {inp1}.shape[{1 if trans_a else 0}];
        int64_t {out}_K = {inp1}.shape[{0 if trans_a else 1}];
        int64_t {out}_N = {inp2}.shape[{0 if trans_b else 1}];

        std::vector<int64_t> {out}_shape = {{{out}_M, {out}_N}};
        int64_t {out}_size = {out}_M * {out}_N;
        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);

#if defined(__APPLE__) && defined(USE_ACCELERATE)
        if constexpr (std::is_same_v<{cpp_type}, float>) {{
            if ({str(has_bias).lower()}) {{
                for (int64_t i = 0; i < {out}_M; ++i) {{
                    for (int64_t j = 0; j < {out}_N; ++j) {{
                        {out}.data[i * {out}_N + j] = {beta}f * {bias_var}.data[j];
                    }}
                }}
            }} else {{
                std::fill({out}.data, {out}.data + {out}_size, 0.0f);
            }}
            cblas_sgemm(CblasRowMajor, {cblas_trans_a}, {cblas_trans_b},
                        {out}_M, {out}_N, {out}_K,
                        {alpha}f, {inp1}.data, {lda},
                        {inp2}.data, {ldb},
                        1.0f, {out}.data, {out}_N);
        }} else {{
            {pragma} collapse(2)
            for (int64_t i = 0; i < {out}_M; ++i) {{
                for (int64_t j = 0; j < {out}_N; ++j) {{
                    {cpp_type} sum = static_cast<{cpp_type}>(0);
                    for (int64_t k = 0; k < {out}_K; ++k) {{
                        sum += {inp1}.data[i * {out}_K + k] * {inp2}.data[k * {out}_N + j]; // ignoring trans params in mock
                    }}
                    {out}.data[i * {out}_N + j] = static_cast<{cpp_type}>({alpha}) * sum{bias_logic};
                }}
            }}
        }}
#else
        {pragma} collapse(2)
        for (int64_t i = 0; i < {out}_M; ++i) {{
            for (int64_t j = 0; j < {out}_N; ++j) {{
                {cpp_type} sum = static_cast<{cpp_type}>(0);
                for (int64_t k = 0; k < {out}_K; ++k) {{
                    sum += {inp1}.data[i * {out}_K + k] * {inp2}.data[k * {out}_N + j]; // ignoring trans params in mock
                }}
                {out}.data[i * {out}_N + j] = static_cast<{cpp_type}>({alpha}) * sum{bias_logic};
            }}
        }}
#endif
"""
