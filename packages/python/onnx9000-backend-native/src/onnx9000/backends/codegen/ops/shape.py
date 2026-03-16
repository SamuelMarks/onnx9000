"""
C++ Code Generation Utilities

Translates ONNX operations to equivalent C++ bindings and memory buffers.
"""

from onnx9000.backends.codegen.generator import Generator
from onnx9000.backends.codegen.utils import get_omp_pragma
from onnx9000.core.ir import Node
from onnx9000.core.registry import global_registry as registry


@registry.register_op("Reshape")
def generate_reshape(node: Node, ctx: Generator) -> str:
    """Implements the generate_reshape method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    return f"\n        // Reshape\n        std::vector<int64_t> {out}_shape = {shape_str};\n        \n        int64_t in_size = 1;\n        for (auto d : {in_obj}.shape) in_size *= d;\n\n        int64_t known_out_size = 1;\n        int64_t neg_idx = -1;\n        for (size_t i = 0; i < {out}_shape.size(); ++i) {{\n            if ({out}_shape[i] == -1) {{\n                neg_idx = i;\n            }} else if ({out}_shape[i] == 0) {{\n                {out}_shape[i] = {in_obj}.shape[i];\n                known_out_size *= {out}_shape[i];\n            }} else {{\n                known_out_size *= {out}_shape[i];\n            }}\n        }}\n\n        if (neg_idx != -1) {{\n            {out}_shape[neg_idx] = in_size / known_out_size;\n        }}\n\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {out}_size *= d;\n\n        if ({out}_size <= 0) {out}_size = in_size;\n\n        if (reinterpret_cast<void*>({in_obj}.data) != reinterpret_cast<void*>(_arena[{buffer_idx}].data())) {{\n            _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n            std::copy({in_obj}.data, {in_obj}.data + in_size, reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()));\n        }} else {{\n            // It IS the same pointer (inplace), but we might still need to update the arena size explicitly if out_size > old_size?\n            // Actually out_size == in_size for reshape ALWAYS, so no need to resize if in-place!\n        }}\n\n        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n    "


@registry.register_op("Flatten")
def generate_flatten(node: Node, ctx: Generator) -> str:
    """Implements the generate_flatten method or operation."""
    return generate_reshape(node, ctx)


@registry.register_op("Squeeze")
def generate_squeeze(node: Node, ctx: Generator) -> str:
    """Implements the generate_squeeze method or operation."""
    return generate_reshape(node, ctx)


@registry.register_op("Unsqueeze")
def generate_unsqueeze(node: Node, ctx: Generator) -> str:
    """Implements the generate_unsqueeze method or operation."""
    return generate_reshape(node, ctx)


@registry.register_op("CastLike")
def generate_cast_like(node: Node, ctx: Generator) -> str:
    """Implements the generate_cast_like method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    ctx.graph.tensors[node.inputs[1]]
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    pragma = get_omp_pragma(f"{inp}.size()")
    return f"\n        // CastLike\n        _arena[{buffer_idx}].resize({inp}.size() * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {inp}.shape);\n        {pragma}\n        for (int64_t i = 0; i < {inp}.size(); ++i) {{\n            {out}.data[i] = static_cast<{cpp_type}>({inp}.data[i]);\n        }}\n"


@registry.register_op("Cast")
def generate_cast(node: Node, ctx: Generator) -> str:
    """Implements the generate_cast method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    pragma = get_omp_pragma(f"{inp}.size()")
    return f"\n        // Cast\n        _arena[{buffer_idx}].resize({inp}.size() * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {inp}.shape);\n        {pragma}\n        for (int64_t i = 0; i < {inp}.size(); ++i) {{\n            {out}.data[i] = static_cast<{cpp_type}>({inp}.data[i]);\n        }}\n"


@registry.register_op("Expand")
def generate_expand(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Expand operator."""
    return generate_reshape(node, ctx)
