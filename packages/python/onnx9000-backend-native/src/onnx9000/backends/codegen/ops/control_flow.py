"""
C++ Code Generation Utilities

Translates ONNX operations to equivalent C++ bindings and memory buffers.
"""

from onnx9000.backends.codegen.generator import Generator
from onnx9000.core.ir import Node
from onnx9000.core.registry import global_registry as registry


@registry.register_op("If")
def generate_if(node: Node, generator_context: Generator) -> str:
    """Implements the generate_if method or operation."""
    cond_var = generator_context.get_tensor_name(node.inputs[0])
    out_blocks = []
    out_blocks.append(
        f"\n        // If\n        if ({cond_var}.data[0] != 0.0f) {{\n            // Execute then branch graph\n        }} else {{\n            // Execute else branch graph\n        }}\n    "
    )
    for _idx, out_name in enumerate(node.outputs):
        out = generator_context.get_tensor_name(out_name)
        tensor_info = generator_context.graph.tensors[out_name]
        buffer_idx = tensor_info.buffer_id
        cpp_type = "float"
        if tensor_info.dtype is not None:
            from onnx9000.core.dtypes import to_cpp_type

            cpp_type = to_cpp_type(tensor_info.dtype)
        out_blocks.append(
            f"\n        // Dummy output allocation for If\n        _arena[{buffer_idx}].resize(1 * sizeof({cpp_type})); // MOCK SIZE\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {{1}});\n        std::fill({out}.data, {out}.data + 1, static_cast<{cpp_type}>(0));\n        "
        )
    return "\n".join(out_blocks)


@registry.register_op("Loop")
def generate_loop(node: Node, generator_context: Generator) -> str:
    """Implements the generate_loop method or operation."""
    max_trip_count = generator_context.get_tensor_name(node.inputs[0])
    cond = generator_context.get_tensor_name(node.inputs[1])
    out_blocks = []
    out_blocks.append(
        f"\n        // Loop\n        int64_t trip_count = static_cast<int64_t>({max_trip_count}.data[0]);\n        bool keep_going = ({cond}.data[0] != 0.0f);\n\n        for (int64_t i = 0; i < trip_count && keep_going; ++i) {{\n            // Execute body subgraph\n\n            // Re-evaluate keep_going from subgraph outputs\n        }}\n    "
    )
    for _idx, out_name in enumerate(node.outputs):
        out = generator_context.get_tensor_name(out_name)
        tensor_info = generator_context.graph.tensors[out_name]
        buffer_idx = tensor_info.buffer_id
        cpp_type = "float"
        if tensor_info.dtype is not None:
            from onnx9000.core.dtypes import to_cpp_type

            cpp_type = to_cpp_type(tensor_info.dtype)
        out_blocks.append(
            f"\n        // Dummy output allocation for Loop\n        _arena[{buffer_idx}].resize(1 * sizeof({cpp_type})); // MOCK SIZE\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {{1}});\n        std::fill({out}.data, {out}.data + 1, static_cast<{cpp_type}>(0));\n        "
        )
    return "\n".join(out_blocks)
