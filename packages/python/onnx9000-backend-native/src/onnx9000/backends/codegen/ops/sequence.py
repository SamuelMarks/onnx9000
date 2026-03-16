"""Sequence operators."""

from onnx9000.backends.codegen.generator import Generator
from onnx9000.core.ir import Node
from onnx9000.core.registry import global_registry as registry


@registry.register_op("SequenceEmpty")
def generate_sequence_empty(node: Node, ctx: Generator) -> str:
    """Implements the generate_sequence_empty method or operation."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    return f"\n        // SequenceEmpty (Mock)\n        _arena[{buffer_idx}].resize(1 * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {{1}});\n        std::fill({out}.data, {out}.data + 1, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("SequenceErase")
def generate_sequence_erase(node: Node, ctx: Generator) -> str:
    """Implements the generate_sequence_erase method or operation."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    return f"\n        // SequenceErase (Mock)\n        _arena[{buffer_idx}].resize(1 * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {{1}});\n        std::fill({out}.data, {out}.data + 1, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("SequenceInsert")
def generate_sequence_insert(node: Node, ctx: Generator) -> str:
    """Implements the generate_sequence_insert method or operation."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    return f"\n        // SequenceInsert (Mock)\n        _arena[{buffer_idx}].resize(1 * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {{1}});\n        std::fill({out}.data, {out}.data + 1, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("SequenceLength")
def generate_sequence_length(node: Node, ctx: Generator) -> str:
    """Implements the generate_sequence_length method or operation."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    return f"\n        // SequenceLength (Mock)\n        _arena[{buffer_idx}].resize(1 * sizeof(int64_t));\n        onnx9000::Tensor<int64_t> {out}(reinterpret_cast<int64_t*>(_arena[{buffer_idx}].data()), {{1}});\n        {out}.data[0] = 0; // Mock empty\n    "


@registry.register_op("SequenceMap")
def generate_sequence_map(node: Node, ctx: Generator) -> str:
    """Implements the generate_sequence_map method or operation."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    return f"\n        // SequenceMap (Mock)\n        _arena[{buffer_idx}].resize(1 * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {{1}});\n        std::fill({out}.data, {out}.data + 1, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("ConcatFromSequence")
def generate_concat_from_sequence(node: Node, ctx: Generator) -> str:
    """Implements the generate_concat_from_sequence method or operation."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    return f"\n        // ConcatFromSequence (Mock)\n        std::vector<int64_t> {out}_shape = {{1}}; // Fallback shape if negative dimensions leak\n        int64_t {out}_size = 1;\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("SplitToSequence")
def generate_split_to_sequence(node: Node, ctx: Generator) -> str:
    """Implements the generate_split_to_sequence method or operation."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    return f"\n        // SplitToSequence (Mock)\n        _arena[{buffer_idx}].resize(1 * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {{1}});\n        std::fill({out}.data, {out}.data + 1, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("SequenceConstruct")
def generate_sequence_construct(node: Node, ctx: Generator) -> str:
    """Implements the generate_sequence_construct method or operation."""
    ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    return f"\n        // SequenceConstruct (Mock)\n        _arena[{buffer_idx}].resize(1 * sizeof({cpp_type}));\n"


@registry.register_op("SequenceAt")
def generate_sequence_at(node: Node, ctx: Generator) -> str:
    """Implements the generate_sequence_at method or operation."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    return f"\n        // SequenceAt (Mock)\n        std::vector<int64_t> {out}_shape = {{1}}; // Fallback shape if negative dimensions leak\n        int64_t {out}_size = 1;\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));\n"
