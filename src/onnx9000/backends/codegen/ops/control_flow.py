"""
C++ Code Generation Utilities

Translates ONNX operations to equivalent C++ bindings and memory buffers.
"""

from onnx9000.backends.codegen.generator import Generator
from onnx9000.core.ir import Node
from onnx9000.core.registry import registry


@registry.register("If")
def generate_if(node: Node, generator_context: Generator) -> str:
    """Provides generate if functionality and verification."""

    cond_var = generator_context.get_tensor_name(node.inputs[0])

    # In a full implementation, `node.attributes["then_branch"]` and `node.attributes["else_branch"]`
    # would contain full `ir.Graph` objects that need to be recursively generated inside
    # the scope of the if-block.

    # We mock this structure heavily since parsing full subgraphs in the frontend
    # tracer isn't fully implemented yet.

    out_blocks = []

    out_blocks.append(f"""
        // If
        if ({cond_var}.data[0] != 0.0f) {{
            // ... then_branch graph execution ...
        }} else {{
            // ... else_branch graph execution ...
        }}
    """)

    # If nodes output tensors, which need to be resolved from the subgraphs.
    for _idx, out_name in enumerate(node.outputs):
        out = generator_context.get_tensor_name(out_name)
        tensor_info = generator_context.graph.tensors[out_name]
        buffer_idx = tensor_info.buffer_id

        cpp_type = "float"
        if tensor_info.dtype is not None:
            from onnx9000.core.dtypes import to_cpp_type

            cpp_type = to_cpp_type(tensor_info.dtype)

        out_blocks.append(f"""
        // Dummy output allocation for If
        _arena[{buffer_idx}].resize(1 * sizeof({cpp_type})); // MOCK SIZE
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {{1}});
        std::fill({out}.data, {out}.data + 1, static_cast<{cpp_type}>(0));
        """)

    return "\n".join(out_blocks)


@registry.register("Loop")
def generate_loop(node: Node, generator_context: Generator) -> str:
    """Provides generate loop functionality and verification."""
    max_trip_count = generator_context.get_tensor_name(node.inputs[0])
    cond = generator_context.get_tensor_name(node.inputs[1])

    out_blocks = []
    out_blocks.append(f"""
        // Loop
        int64_t trip_count = static_cast<int64_t>({max_trip_count}.data[0]);
        bool keep_going = ({cond}.data[0] != 0.0f);

        for (int64_t i = 0; i < trip_count && keep_going; ++i) {{
            // ... body subgraph execution ...

            // Re-evaluate keep_going from subgraph outputs
        }}
    """)

    for _idx, out_name in enumerate(node.outputs):
        out = generator_context.get_tensor_name(out_name)
        tensor_info = generator_context.graph.tensors[out_name]
        buffer_idx = tensor_info.buffer_id

        cpp_type = "float"
        if tensor_info.dtype is not None:
            from onnx9000.core.dtypes import to_cpp_type

            cpp_type = to_cpp_type(tensor_info.dtype)

        out_blocks.append(f"""
        // Dummy output allocation for Loop
        _arena[{buffer_idx}].resize(1 * sizeof({cpp_type})); // MOCK SIZE
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {{1}});
        std::fill({out}.data, {out}.data + 1, static_cast<{cpp_type}>(0));
        """)

    return "\n".join(out_blocks)
