"""Sequence operators."""

from onnx9000.backends.codegen.generator import Generator
from onnx9000.core.ir import Node
from onnx9000.core.registry import registry


@registry.register("SequenceEmpty")
def generate_sequence_empty(node: Node, ctx: Generator) -> str:
    """Provides generate sequence empty functionality and verification."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    return f"""
        // SequenceEmpty (Mock)
        _arena[{buffer_idx}].resize(1 * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {{1}});
        std::fill({out}.data, {out}.data + 1, static_cast<{cpp_type}>(0));
    """


@registry.register("SequenceErase")
def generate_sequence_erase(node: Node, ctx: Generator) -> str:
    """Provides generate sequence erase functionality and verification."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    return f"""
        // SequenceErase (Mock)
        _arena[{buffer_idx}].resize(1 * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {{1}});
        std::fill({out}.data, {out}.data + 1, static_cast<{cpp_type}>(0));
    """


@registry.register("SequenceInsert")
def generate_sequence_insert(node: Node, ctx: Generator) -> str:
    """Provides generate sequence insert functionality and verification."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    return f"""
        // SequenceInsert (Mock)
        _arena[{buffer_idx}].resize(1 * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {{1}});
        std::fill({out}.data, {out}.data + 1, static_cast<{cpp_type}>(0));
    """


@registry.register("SequenceLength")
def generate_sequence_length(node: Node, ctx: Generator) -> str:
    """Provides generate sequence length functionality and verification."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    # Needs to return an INT64 scalar
    return f"""
        // SequenceLength (Mock)
        _arena[{buffer_idx}].resize(1 * sizeof(int64_t));
        onnx9000::Tensor<int64_t> {out}(reinterpret_cast<int64_t*>(_arena[{buffer_idx}].data()), {{1}});
        {out}.data[0] = 0; // Mock empty
    """


@registry.register("SequenceMap")
def generate_sequence_map(node: Node, ctx: Generator) -> str:
    """Provides generate sequence map functionality and verification."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    return f"""
        // SequenceMap (Mock)
        _arena[{buffer_idx}].resize(1 * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {{1}});
        std::fill({out}.data, {out}.data + 1, static_cast<{cpp_type}>(0));
    """


@registry.register("ConcatFromSequence")
def generate_concat_from_sequence(node: Node, ctx: Generator) -> str:
    """Provides generate concat from sequence functionality and verification."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    return f"""
        // ConcatFromSequence (Mock)
        std::vector<int64_t> {out}_shape = {{1}}; // Fallback shape if negative dimensions leak
        int64_t {out}_size = 1;

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));
    """


@registry.register("SplitToSequence")
def generate_split_to_sequence(node: Node, ctx: Generator) -> str:
    """Provides generate split to sequence functionality and verification."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    return f"""
        // SplitToSequence (Mock)
        _arena[{buffer_idx}].resize(1 * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {{1}});
        std::fill({out}.data, {out}.data + 1, static_cast<{cpp_type}>(0));
    """


@registry.register("SequenceConstruct")
def generate_sequence_construct(node: Node, ctx: Generator) -> str:
    """Provides generate sequence construct functionality and verification."""
    out = ctx.get_tensor_name(node.outputs[0])

    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    return f"""
        // SequenceConstruct (Mock)
        _arena[{buffer_idx}].resize(1 * sizeof({cpp_type}));
"""


@registry.register("SequenceAt")
def generate_sequence_at(node: Node, ctx: Generator) -> str:
    """Provides generate sequence at functionality and verification."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    return f"""
        // SequenceAt (Mock)
        std::vector<int64_t> {out}_shape = {{1}}; // Fallback shape if negative dimensions leak
        int64_t {out}_size = 1;

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));
"""
