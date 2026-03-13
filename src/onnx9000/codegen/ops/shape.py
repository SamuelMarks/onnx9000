"""
C++ Code Generation Utilities

Translates ONNX operations to equivalent C++ bindings and memory buffers.
"""

from onnx9000.codegen.generator import Generator
from onnx9000.codegen.utils import get_omp_pragma
from onnx9000.ir import Node
from onnx9000.registry import registry


@registry.register("Reshape")
def generate_reshape(node: Node, ctx: Generator) -> str:
    """generate_reshape docstring."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and not ctx.graph.tensors[node.inputs[0]].is_initializer
        else inp
    )

    return f"""
        // Reshape
        std::vector<int64_t> {out}_shape = {shape_str};
        
        int64_t in_size = 1;
        for (auto d : {in_obj}.shape) in_size *= d;

        int64_t known_out_size = 1;
        int64_t neg_idx = -1;
        for (size_t i = 0; i < {out}_shape.size(); ++i) {{
            if ({out}_shape[i] == -1) {{
                neg_idx = i;
            }} else if ({out}_shape[i] == 0) {{
                {out}_shape[i] = {in_obj}.shape[i];
                known_out_size *= {out}_shape[i];
            }} else {{
                known_out_size *= {out}_shape[i];
            }}
        }}

        if (neg_idx != -1) {{
            {out}_shape[neg_idx] = in_size / known_out_size;
        }}

        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {out}_size *= d;

        if ({out}_size <= 0) {out}_size = in_size;

        if (reinterpret_cast<void*>({in_obj}.data) != reinterpret_cast<void*>(_arena[{buffer_idx}].data())) {{
            _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
            std::copy({in_obj}.data, {in_obj}.data + in_size, reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()));
        }} else {{
            // It IS the same pointer (inplace), but we might still need to update the arena size explicitly if out_size > old_size?
            // Actually out_size == in_size for reshape ALWAYS, so no need to resize if in-place!
        }}

        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
    """


@registry.register("Flatten")
def generate_flatten(node: Node, ctx: Generator) -> str:
    """generate_flatten docstring."""

    return generate_reshape(node, ctx)  # Conceptually similar for mock


@registry.register("Squeeze")
def generate_squeeze(node: Node, ctx: Generator) -> str:
    """generate_squeeze docstring."""

    return generate_reshape(node, ctx)


@registry.register("Unsqueeze")
def generate_unsqueeze(node: Node, ctx: Generator) -> str:
    """generate_unsqueeze docstring."""

    return generate_reshape(node, ctx)


@registry.register("CastLike")
def generate_cast_like(node: Node, ctx: Generator) -> str:
    """generate_cast_like docstring."""
    inp = ctx.get_tensor_name(node.inputs[0])
    # Target tensor dtype
    target_tensor_info = ctx.graph.tensors[node.inputs[1]]
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    pragma = get_omp_pragma(f"{inp}.size()")

    return f"""
        // CastLike
        _arena[{buffer_idx}].resize({inp}.size() * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {inp}.shape);
        {pragma}
        for (int64_t i = 0; i < {inp}.size(); ++i) {{
            {out}.data[i] = static_cast<{cpp_type}>({inp}.data[i]);
        }}
"""


@registry.register("Cast")
def generate_cast(node: Node, ctx: Generator) -> str:
    """generate_cast docstring."""

    # Requires type conversion map
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    pragma = get_omp_pragma(f"{inp}.size()")

    return f"""
        // Cast
        _arena[{buffer_idx}].resize({inp}.size() * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {inp}.shape);
        {pragma}
        for (int64_t i = 0; i < {inp}.size(); ++i) {{
            {out}.data[i] = static_cast<{cpp_type}>({inp}.data[i]);
        }}
"""


@registry.register("Expand")
def generate_expand(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Expand operator."""
    return generate_reshape(node, ctx)
