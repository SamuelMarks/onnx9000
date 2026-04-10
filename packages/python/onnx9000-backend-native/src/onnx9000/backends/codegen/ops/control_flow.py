"""C++ Code Generation Utilities.

Translates ONNX operations to equivalent C++ bindings and memory buffers.
"""

from onnx9000.core.ir import Node
from onnx9000.core.registry import global_registry as registry


@registry.register_op("", "If")
def generate_if(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implement the generate_if method or operation."""
    cond_var = ctx.get_tensor_name(node.inputs[0])

    then_graph = node.attributes.get("then_branch")
    else_graph = node.attributes.get("else_branch")

    then_op_codes = []
    if then_graph:
        for then_node in then_graph.nodes:
            op_gen = registry.get_op(getattr(then_node, "domain", ""), then_node.op_type)
            then_op_codes.append(op_gen(then_node, ctx))

    else_op_codes = []
    if else_graph:
        for else_node in else_graph.nodes:
            op_gen = registry.get_op(getattr(else_node, "domain", ""), else_node.op_type)
            else_op_codes.append(op_gen(else_node, ctx))

    then_code = "\\n".join(then_op_codes)
    else_code = "\\n".join(else_op_codes)

    out_blocks = []
    out_blocks.append(
        f"\\n        // If\\n        if ({cond_var}.data[0] != 0.0f) {{\n{then_code}\\n        }} else {{\n{else_code}\\n        }}\\n    "
    )
    for _idx, out_name in enumerate(node.outputs):
        out = ctx.get_tensor_name(out_name)
        tensor_info = ctx.graph.tensors[out_name]
        offset = ctx.tensor_offsets.get(node.outputs[0], 0)
        cpp_type = "float"
        if tensor_info.dtype is not None:
            from onnx9000.core.dtypes import to_cpp_type

            cpp_type = to_cpp_type(tensor_info.dtype)
        out_blocks.append(
            f"\\n        // Dummy output allocation for If\\n        /* preallocated */ // MOCK SIZE\\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {{1}});\\n        std::fill({out}.data, {out}.data + 1, static_cast<{cpp_type}>(0));\\n        "
        )
    return "\\n".join(out_blocks)


@registry.register_op("", "Loop")
def generate_loop(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implement the generate_loop method or operation."""
    max_trip_count = (
        ctx.get_tensor_name(node.inputs[0]) if len(node.inputs) > 0 and node.inputs[0] else None
    )
    cond = ctx.get_tensor_name(node.inputs[1]) if len(node.inputs) > 1 and node.inputs[1] else None

    body_graph = node.attributes.get("body")
    body_op_codes = []
    if body_graph:
        for body_node in body_graph.nodes:
            op_gen = registry.get_op(getattr(body_node, "domain", ""), body_node.op_type)
            body_op_codes.append(op_gen(body_node, ctx))

    body_code = "\\n".join(body_op_codes)

    out_blocks = []

    trip_decl = (
        f"int64_t trip_count = static_cast<int64_t>({max_trip_count}.data[0]);"
        if max_trip_count
        else "int64_t trip_count = INT64_MAX;"
    )
    cond_decl = (
        f"bool keep_going = ({cond}.data[0] != 0.0f);" if cond else "bool keep_going = true;"
    )

    out_blocks.append(
        f"\\n        // Loop\\n        {trip_decl}\\n        {cond_decl}\\n\\n        for (int64_t i = 0; i < trip_count && keep_going; ++i) {{\n{body_code}\\n        }}\\n    "
    )
    for _idx, out_name in enumerate(node.outputs):
        out = ctx.get_tensor_name(out_name)
        tensor_info = ctx.graph.tensors[out_name]
        offset = ctx.tensor_offsets.get(node.outputs[0], 0)
        cpp_type = "float"
        if tensor_info.dtype is not None:
            from onnx9000.core.dtypes import to_cpp_type

            cpp_type = to_cpp_type(tensor_info.dtype)
        out_blocks.append(
            f"\\n        // Dummy output allocation for Loop\\n        /* preallocated */ // MOCK SIZE\\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {{1}});\\n        std::fill({out}.data, {out}.data + 1, static_cast<{cpp_type}>(0));\\n        "
        )
    return "\\n".join(out_blocks)
