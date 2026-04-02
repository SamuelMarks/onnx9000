"""C++ Code Generation Utilities.

Translates ONNX operations to equivalent C++ bindings and memory buffers.
"""

from onnx9000.backends.codegen.utils import get_omp_pragma
from onnx9000.core.ir import Node
from onnx9000.core.registry import global_registry as registry


@registry.register_op("Reshape")
def generate_reshape(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implement the generate_reshape method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    return f"""
        // Reshape
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = {inp}.size();
        
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {out}_shape);
        if (reinterpret_cast<void*>({inp}.data) != reinterpret_cast<void*>({out}.data)) {{
            std::copy({inp}.data, {inp}.data + {out}_size, {out}.data);
        }}
    """


@registry.register_op("Flatten")
def generate_flatten(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implement the generate_flatten method or operation."""
    return generate_reshape(node, ctx)


@registry.register_op("Squeeze")
def generate_squeeze(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implement the generate_squeeze method or operation."""
    return generate_reshape(node, ctx)


@registry.register_op("Unsqueeze")
def generate_unsqueeze(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implement the generate_unsqueeze method or operation."""
    return generate_reshape(node, ctx)


@registry.register_op("CastLike")
def generate_cast_like(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implement the generate_cast_like method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    pragma = get_omp_pragma(f"{inp}.size()")
    return f"""
        // CastLike
        /* preallocated */
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {inp}.shape);
        {pragma}
        for (int64_t i = 0; i < {inp}.size(); ++i) {{
            {out}.data[i] = static_cast<{cpp_type}>({inp}.data[i]);
        }}
"""


@registry.register_op("Cast")
def generate_cast(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implement the generate_cast method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    pragma = get_omp_pragma(f"{inp}.size()")
    return f"""
        // Cast
        /* preallocated */
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {inp}.shape);
        {pragma}
        for (int64_t i = 0; i < {inp}.size(); ++i) {{
            {out}.data[i] = static_cast<{cpp_type}>({inp}.data[i]);
        }}
"""


@registry.register_op("Expand")
def generate_expand(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Generate the code implementation for the Expand operator."""
    return generate_reshape(node, ctx)
