"""
C++ Code Generation Utilities

Translates ONNX operations to equivalent C++ bindings and memory buffers.
"""

from onnx9000.backends.codegen.generator import Generator
from onnx9000.backends.codegen.utils import get_omp_pragma
from onnx9000.core.ir import Node
from onnx9000.core.registry import global_registry as registry


@registry.register_op("MatMul")
def generate_matmul(node: Node, generator_context: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_matmul method or operation."""
    inp1 = generator_context.get_tensor_name(node.inputs[0])
    inp2 = generator_context.get_tensor_name(node.inputs[1])
    out = generator_context.get_tensor_name(node.outputs[0])
    tensor_info = generator_context.graph.tensors[node.outputs[0]]
    offset = generator_context.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    pragma = get_omp_pragma(f"{out}_size", extra="collapse(2)")
    return f"\n        // MatMul (Cache-blocked / Tiled)\n        int64_t {out}_M = {inp1}.shape[0];\n        int64_t {out}_K = {inp1}.shape[1];\n        int64_t {out}_N = {inp2}.shape[1];\n\n        std::vector<int64_t> {out}_shape = {{{out}_M, {out}_N}};\n        int64_t {out}_size = {out}_M * {out}_N;\n        /* preallocated */\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {out}_shape);\n\n        #if defined(__APPLE__) && defined(USE_ACCELERATE)\n        if constexpr (std::is_same_v<{cpp_type}, float>) {{\n            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,\n                        {out}_M, {out}_N, {out}_K,\n                        1.0f, {inp1}.data, {out}_K,\n                        {inp2}.data, {out}_N,\n                        0.0f, {out}.data, {out}_N);\n        }} else {{\n            // Cache-blocked fallback\n            int64_t BLOCK_SIZE = 64;\n            for (int64_t i = 0; i < {out}_M * {out}_N; ++i) {out}.data[i] = static_cast<{cpp_type}>(0);\n            {pragma}\n            for (int64_t i0 = 0; i0 < {out}_M; i0 += BLOCK_SIZE) {{\n                for (int64_t k0 = 0; k0 < {out}_K; k0 += BLOCK_SIZE) {{\n                    for (int64_t j0 = 0; j0 < {out}_N; j0 += BLOCK_SIZE) {{\n                        for (int64_t i = i0; i < std::min({out}_M, i0 + BLOCK_SIZE); ++i) {{\n                            for (int64_t k = k0; k < std::min({out}_K, k0 + BLOCK_SIZE); ++k) {{\n                                {cpp_type} a_val = {inp1}.data[i * {out}_K + k];\n                                for (int64_t j = j0; j < std::min({out}_N, j0 + BLOCK_SIZE); ++j) {{\n                                    {out}.data[i * {out}_N + j] += a_val * {inp2}.data[k * {out}_N + j];\n                                }}\n                            }}\n                        }}\n                    }}\n                }}\n            }}\n        }}\n        #else\n        // Cache-blocked naive implementation\n        int64_t BLOCK_SIZE = 64;\n        for (int64_t i = 0; i < {out}_M * {out}_N; ++i) {out}.data[i] = static_cast<{cpp_type}>(0);\n        {pragma}\n        for (int64_t i0 = 0; i0 < {out}_M; i0 += BLOCK_SIZE) {{\n            for (int64_t k0 = 0; k0 < {out}_K; k0 += BLOCK_SIZE) {{\n                for (int64_t j0 = 0; j0 < {out}_N; j0 += BLOCK_SIZE) {{\n                    for (int64_t i = i0; i < std::min({out}_M, i0 + BLOCK_SIZE); ++i) {{\n                        for (int64_t k = k0; k < std::min({out}_K, k0 + BLOCK_SIZE); ++k) {{\n                            {cpp_type} a_val = {inp1}.data[i * {out}_K + k];\n                            for (int64_t j = j0; j < std::min({out}_N, j0 + BLOCK_SIZE); ++j) {{\n                                {out}.data[i * {out}_N + j] += a_val * {inp2}.data[k * {out}_N + j];\n                            }}\n                        }}\n                    }}\n                }}\n            }}\n        }}\n        #endif\n        "


@registry.register_op("Gemm")
def generate_gemm(node: Node, generator_context: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_gemm method or operation."""
    inp1 = generator_context.get_tensor_name(node.inputs[0])
    inp2 = generator_context.get_tensor_name(node.inputs[1])
    has_bias = len(node.inputs) > 2
    bias_var = generator_context.get_tensor_name(node.inputs[2]) if has_bias else None
    out = generator_context.get_tensor_name(node.outputs[0])
    tensor_info = generator_context.graph.tensors[node.outputs[0]]
    offset = generator_context.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    alpha = node.attributes.get("alpha", 1.0)
    beta = node.attributes.get("beta", 1.0)
    trans_a = node.attributes.get("trans_a", 0)
    trans_b = node.attributes.get("trans_b", 0)
    pragma = get_omp_pragma(f"{out}_size", extra="collapse(2)")
    bias_logic = f" + static_cast<{cpp_type}>({beta}) * {bias_var}.data[j]" if has_bias else ""
    cblas_trans_a = "CblasTrans" if trans_a else "CblasNoTrans"
    cblas_trans_b = "CblasTrans" if trans_b else "CblasNoTrans"
    lda = f"{out}_M" if trans_a else f"{out}_K"
    ldb = f"{out}_K" if trans_b else f"{out}_N"
    return f"\n        // Gemm (Fused MatMul + Add)\n        int64_t {out}_M = {inp1}.shape[{(1 if trans_a else 0)}];\n        int64_t {out}_K = {inp1}.shape[{(0 if trans_a else 1)}];\n        int64_t {out}_N = {inp2}.shape[{(0 if trans_b else 1)}];\n\n        std::vector<int64_t> {out}_shape = {{{out}_M, {out}_N}};\n        int64_t {out}_size = {out}_M * {out}_N;\n        /* preallocated */\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {out}_shape);\n\n#if defined(__APPLE__) && defined(USE_ACCELERATE)\n        if constexpr (std::is_same_v<{cpp_type}, float>) {{\n            if ({str(has_bias).lower()}) {{\n                for (int64_t i = 0; i < {out}_M; ++i) {{\n                    for (int64_t j = 0; j < {out}_N; ++j) {{\n                        {out}.data[i * {out}_N + j] = {beta}f * {bias_var}.data[j];\n                    }}\n                }}\n            }} else {{\n                std::fill({out}.data, {out}.data + {out}_size, 0.0f);\n            }}\n            cblas_sgemm(CblasRowMajor, {cblas_trans_a}, {cblas_trans_b},\n                        {out}_M, {out}_N, {out}_K,\n                        {alpha}f, {inp1}.data, {lda},\n                        {inp2}.data, {ldb},\n                        1.0f, {out}.data, {out}_N);\n        }} else {{\n            {pragma}\n            for (int64_t i = 0; i < {out}_M; ++i) {{\n                for (int64_t j = 0; j < {out}_N; ++j) {{\n                    {cpp_type} sum = static_cast<{cpp_type}>(0);\n                    for (int64_t k = 0; k < {out}_K; ++k) {{\n                        sum += {inp1}.data[i * {out}_K + k] * {inp2}.data[k * {out}_N + j]; // ignoring trans params in mock\n                    }}\n                    {out}.data[i * {out}_N + j] = static_cast<{cpp_type}>({alpha}) * sum{bias_logic};\n                }}\n            }}\n        }}\n#else\n        {pragma}\n        for (int64_t i = 0; i < {out}_M; ++i) {{\n            for (int64_t j = 0; j < {out}_N; ++j) {{\n                {cpp_type} sum = static_cast<{cpp_type}>(0);\n                for (int64_t k = 0; k < {out}_K; ++k) {{\n                    sum += {inp1}.data[i * {out}_K + k] * {inp2}.data[k * {out}_N + j]; // ignoring trans params in mock\n                }}\n                {out}.data[i * {out}_N + j] = static_cast<{cpp_type}>({alpha}) * sum{bias_logic};\n            }}\n        }}\n#endif\n"
