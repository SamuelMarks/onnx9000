"""Sequence operators."""

from onnx9000.backends.codegen.generator import Generator
from onnx9000.core.dtypes import to_cpp_type
from onnx9000.core.ir import Node
from onnx9000.core.registry import global_registry as registry


@registry.register_op("", "SequenceEmpty")
def generate_sequence_empty(node: Node, ctx: Generator) -> str:
    """Generate target source code for the sequence empty operator.

    Args:
        node: Parameter specifying node (Node).
        ctx: Parameter specifying ctx ('onnx9000.backends.codegen.Generator').

    Returns:
        Result of generate sequence empty (str).
    """
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = to_cpp_type(tensor_info.dtype) if tensor_info.dtype is not None else "float"
    return f"\n        // SequenceEmpty (Mock)\n        /* preallocated */\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {{1}});\n        std::fill({out}.data, {out}.data + 1, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("", "SequenceErase")
def generate_sequence_erase(node: Node, ctx: Generator) -> str:
    """Generate target source code for the sequence erase operator.

    Args:
        node: Parameter specifying node (Node).
        ctx: Parameter specifying ctx ('onnx9000.backends.codegen.Generator').

    Returns:
        Result of generate sequence erase (str).
    """
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = to_cpp_type(tensor_info.dtype) if tensor_info.dtype is not None else "float"
    return f"\n        // SequenceErase (Mock)\n        /* preallocated */\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {{1}});\n        std::fill({out}.data, {out}.data + 1, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("", "SequenceInsert")
def generate_sequence_insert(node: Node, ctx: Generator) -> str:
    """Generate target source code for the sequence insert operator.

    Args:
        node: Parameter specifying node (Node).
        ctx: Parameter specifying ctx ('onnx9000.backends.codegen.Generator').

    Returns:
        Result of generate sequence insert (str).
    """
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = to_cpp_type(tensor_info.dtype) if tensor_info.dtype is not None else "float"
    return f"\n        // SequenceInsert (Mock)\n        /* preallocated */\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {{1}});\n        std::fill({out}.data, {out}.data + 1, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("", "SequenceLength")
def generate_sequence_length(node: Node, ctx: Generator) -> str:
    """Generate target source code for the sequence length operator.

    Args:
        node: Parameter specifying node (Node).
        ctx: Parameter specifying ctx ('onnx9000.backends.codegen.Generator').

    Returns:
        Result of generate sequence length (str).
    """
    out = ctx.get_tensor_name(node.outputs[0])
    ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    return f"\n        // SequenceLength (Mock)\n        /* preallocated */\n        onnx9000::Tensor<int64_t> {out}(reinterpret_cast<int64_t*>((_global_arena.data() + {offset})), {{1}});\n        {out}.data[0] = 0; // Mock empty\n    "


@registry.register_op("", "SequenceMap")
def generate_sequence_map(node: Node, ctx: Generator) -> str:
    """Generate target source code for the sequence map operator.

    Args:
        node: Parameter specifying node (Node).
        ctx: Parameter specifying ctx ('onnx9000.backends.codegen.Generator').

    Returns:
        Result of generate sequence map (str).
    """
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = to_cpp_type(tensor_info.dtype) if tensor_info.dtype is not None else "float"
    return f"\n        // SequenceMap (Mock)\n        /* preallocated */\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {{1}});\n        std::fill({out}.data, {out}.data + 1, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("", "ConcatFromSequence")
def generate_concat_from_sequence(node: Node, ctx: Generator) -> str:
    """Generate target source code for the concat from sequence operator.

    Args:
        node: Parameter specifying node (Node).
        ctx: Parameter specifying ctx ('onnx9000.backends.codegen.Generator').

    Returns:
        Result of generate concat from sequence (str).
    """
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = to_cpp_type(tensor_info.dtype) if tensor_info.dtype is not None else "float"
    "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    return f"\n        // ConcatFromSequence (Mock)\n        std::vector<int64_t> {out}_shape = {{1}}; // Fallback shape if negative dimensions leak\n        int64_t {out}_size = 1;\n\n        /* preallocated */\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {out}_shape);\n        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("", "SplitToSequence")
def generate_split_to_sequence(node: Node, ctx: Generator) -> str:
    """Generate target source code for the split to sequence operator.

    Args:
        node: Parameter specifying node (Node).
        ctx: Parameter specifying ctx ('onnx9000.backends.codegen.Generator').

    Returns:
        Result of generate split to sequence (str).
    """
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = to_cpp_type(tensor_info.dtype) if tensor_info.dtype is not None else "float"
    return f"\n        // SplitToSequence (Mock)\n        /* preallocated */\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {{1}});\n        std::fill({out}.data, {out}.data + 1, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("", "SequenceConstruct")
def generate_sequence_construct(node: Node, ctx: Generator) -> str:
    """Generate target source code for the sequence construct operator.

    Args:
        node: Parameter specifying node (Node).
        ctx: Parameter specifying ctx ('onnx9000.backends.codegen.Generator').

    Returns:
        Result of generate sequence construct (str).
    """
    ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    ctx.tensor_offsets.get(node.outputs[0], 0)
    _ = to_cpp_type(tensor_info.dtype) if tensor_info.dtype is not None else "float"
    return "\n        // SequenceConstruct (Mock)\n        /* preallocated */\n"


@registry.register_op("", "SequenceAt")
def generate_sequence_at(node: Node, ctx: Generator) -> str:
    """Generate target source code for the sequence at operator.

    Args:
        node: Parameter specifying node (Node).
        ctx: Parameter specifying ctx ('onnx9000.backends.codegen.Generator').

    Returns:
        Result of generate sequence at (str).
    """
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = to_cpp_type(tensor_info.dtype) if tensor_info.dtype is not None else "float"
    "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    return f"\n        // SequenceAt (Mock)\n        std::vector<int64_t> {out}_shape = {{1}}; // Fallback shape if negative dimensions leak\n        int64_t {out}_size = 1;\n\n        /* preallocated */\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {out}_shape);\n        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));\n"
