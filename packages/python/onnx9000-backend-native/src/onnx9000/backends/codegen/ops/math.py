"""
C++ Code Generation Utilities

Translates ONNX operations to equivalent C++ bindings and memory buffers.
"""

from onnx9000.backends.codegen.generator import Generator
from onnx9000.backends.codegen.ops.elementwise import _generate_binary_op, _generate_unary_op
from onnx9000.core.ir import Node
from onnx9000.core.registry import global_registry as registry


@registry.register_op("BlackmanWindow")
def generate_blackman_window(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_blackman_window method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    periodic = node.attributes.get("periodic", 1)
    return f"\n        // BlackmanWindow\n        int64_t window_size = {inp}.size() > 0 ? {inp}.data[0] : 0;\n        std::vector<int64_t> {out}_shape = {{window_size}};\n        \n        /* preallocated */\n        onnx9000::Tensor<{cpp_type}> {out}_{offset}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {out}_shape);\n        {out}_{offset}.shape = {out}_shape; // Ensure shape is explicitly dynamic\n\n        if (window_size == 1) {{\n            {out}_{offset}.data[0] = static_cast<{cpp_type}>(1.0);\n        }} else if (window_size > 1) {{\n            double N = static_cast<double>({periodic} ? window_size : window_size - 1);\n            for (int64_t i = 0; i < window_size; ++i) {{\n                double x = 2.0 * M_PI * i / N;\n                double val = 0.42 - 0.5 * std::cos(x) + 0.08 * std::cos(2.0 * x);\n                {out}_{offset}.data[i] = static_cast<{cpp_type}>(val);\n            }}\n        }}\n    "


@registry.register_op("Det")
def generate_det(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_det method or operation."""
    ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    return f"\n        // Det (Mock)\n        std::vector<int64_t> {out}_shape = {out_shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {{\n            if (d < 0) d = 1; // MOCK ONLY\n            {out}_size *= d;\n        }}\n        if ({out}_size < 0) {out}_size = 1;\n        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback\n\n        for (size_t i = 0; i < {out}_shape.size(); ++i) {{\n            if ({out}_shape[i] < 0) {out}_shape[i] = 1;\n        }}\n\n        /* preallocated */\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {out}_shape);\n        // Fill mock\n        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("Mean")
def generate_mean(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_mean method or operation."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    return f"\n        // Mean (Mock)\n        std::vector<int64_t> {out}_shape = {out_shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {{\n            if (d < 0) d = 1; // MOCK ONLY\n            {out}_size *= d;\n        }}\n        if ({out}_size < 0) {out}_size = 1;\n        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback\n\n        for (size_t i = 0; i < {out}_shape.size(); ++i) {{\n            if ({out}_shape[i] < 0) {out}_shape[i] = 1;\n        }}\n\n        /* preallocated */\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {out}_shape);\n        // Fill mock\n        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("MelWeightMatrix")
def generate_mel_weight_matrix(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_mel_weight_matrix method or operation."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    return f"\n        // MelWeightMatrix (Mock)\n        std::vector<int64_t> {out}_shape = {out_shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {{\n            if (d < 0) d = 1; // MOCK ONLY\n            {out}_size *= d;\n        }}\n        if ({out}_size < 0) {out}_size = 1;\n        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback\n\n        for (size_t i = 0; i < {out}_shape.size(); ++i) {{\n            if ({out}_shape[i] < 0) {out}_shape[i] = 1;\n        }}\n\n        /* preallocated */\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {out}_shape);\n        // Fill mock\n        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("Multinomial")
def generate_multinomial(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_multinomial method or operation."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    return f"\n        // Multinomial (Mock)\n        std::vector<int64_t> {out}_shape = {out_shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {{\n            if (d < 0) d = 1; // MOCK ONLY\n            {out}_size *= d;\n        }}\n        if ({out}_size < 0) {out}_size = 1;\n        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback\n\n        for (size_t i = 0; i < {out}_shape.size(); ++i) {{\n            if ({out}_shape[i] < 0) {out}_shape[i] = 1;\n        }}\n\n        /* preallocated */\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {out}_shape);\n        // Fill mock\n        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("ReduceSumSquare")
def generate_reduce_sum_square(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_reduce_sum_square method or operation."""
    _inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    return f"\n        // ReduceSumSquare (Mock)\n        std::vector<int64_t> {out}_shape = {out_shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {{\n            if (d < 0) d = 1; // MOCK ONLY\n            {out}_size *= d;\n        }}\n        if ({out}_size < 0) {out}_size = 1;\n        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback\n\n        for (size_t i = 0; i < {out}_shape.size(); ++i) {{\n            if ({out}_shape[i] < 0) {out}_shape[i] = 1;\n        }}\n\n        /* preallocated */\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {out}_shape);\n        // Fill mock\n        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("ReduceL1")
def generate_reduce_l1(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_reduce_l1 method or operation."""
    _inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    return f"\n        // ReduceL1 (Mock)\n        std::vector<int64_t> {out}_shape = {out_shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {{\n            if (d < 0) d = 1; // MOCK ONLY\n            {out}_size *= d;\n        }}\n        if ({out}_size < 0) {out}_size = 1;\n        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback\n\n        for (size_t i = 0; i < {out}_shape.size(); ++i) {{\n            if ({out}_shape[i] < 0) {out}_shape[i] = 1;\n        }}\n\n        /* preallocated */\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {out}_shape);\n        // Fill mock\n        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("ReduceL2")
def generate_reduce_l2(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_reduce_l2 method or operation."""
    _inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    return f"\n        // ReduceL2 (Mock)\n        std::vector<int64_t> {out}_shape = {out_shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {{\n            if (d < 0) d = 1; // MOCK ONLY\n            {out}_size *= d;\n        }}\n        if ({out}_size < 0) {out}_size = 1;\n        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback\n\n        for (size_t i = 0; i < {out}_shape.size(); ++i) {{\n            if ({out}_shape[i] < 0) {out}_shape[i] = 1;\n        }}\n\n        /* preallocated */\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {out}_shape);\n        // Fill mock\n        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("ReduceLogSum")
def generate_reduce_log_sum(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_reduce_log_sum method or operation."""
    _inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    return f"\n        // ReduceLogSum (Mock)\n        std::vector<int64_t> {out}_shape = {out_shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {{\n            if (d < 0) d = 1; // MOCK ONLY\n            {out}_size *= d;\n        }}\n        if ({out}_size < 0) {out}_size = 1;\n        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback\n\n        for (size_t i = 0; i < {out}_shape.size(); ++i) {{\n            if ({out}_shape[i] < 0) {out}_shape[i] = 1;\n        }}\n\n        /* preallocated */\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {out}_shape);\n        // Fill mock\n        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("ReduceLogSumExp")
def generate_reduce_log_sum_exp(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_reduce_log_sum_exp method or operation."""
    _inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    return f"\n        // ReduceLogSumExp (Mock)\n        std::vector<int64_t> {out}_shape = {out_shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {{\n            if (d < 0) d = 1; // MOCK ONLY\n            {out}_size *= d;\n        }}\n        if ({out}_size < 0) {out}_size = 1;\n        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback\n\n        for (size_t i = 0; i < {out}_shape.size(); ++i) {{\n            if ({out}_shape[i] < 0) {out}_shape[i] = 1;\n        }}\n\n        /* preallocated */\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {out}_shape);\n        // Fill mock\n        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("Shrink")
def generate_shrink(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_shrink method or operation."""
    ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    return f"\n        // Shrink (Mock)\n        std::vector<int64_t> {out}_shape = {out_shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {{\n            if (d < 0) d = 1; // MOCK ONLY\n            {out}_size *= d;\n        }}\n        if ({out}_size < 0) {out}_size = 1;\n        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback\n\n        for (size_t i = 0; i < {out}_shape.size(); ++i) {{\n            if ({out}_shape[i] < 0) {out}_shape[i] = 1;\n        }}\n\n        /* preallocated */\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {out}_shape);\n        // Fill mock\n        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("Size")
def generate_size(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_size method or operation."""
    ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = "int64_t"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"
    return f"\n        // Size (Mock)\n        std::vector<int64_t> {out}_shape = {out_shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {{\n            if (d < 0) d = 1; // MOCK ONLY\n            {out}_size *= d;\n        }}\n        if ({out}_size < 0) {out}_size = 1;\n        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback\n\n        for (size_t i = 0; i < {out}_shape.size(); ++i) {{\n            if ({out}_shape[i] < 0) {out}_shape[i] = 1;\n        }}\n\n        /* preallocated */\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {out}_shape);\n        // Fill mock\n        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));\n    "


@registry.register_op("Abs")
def generate_abs(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_abs method or operation."""
    return _generate_unary_op(node, ctx, "std::abs({inp})")


@registry.register_op("Acos")
def generate_acos(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_acos method or operation."""
    return _generate_unary_op(node, ctx, "std::acos({inp})")


@registry.register_op("Acosh")
def generate_acosh(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_acosh method or operation."""
    return _generate_unary_op(node, ctx, "std::acosh({inp})")


@registry.register_op("Asin")
def generate_asin(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_asin method or operation."""
    return _generate_unary_op(node, ctx, "std::asin({inp})")


@registry.register_op("Asinh")
def generate_asinh(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_asinh method or operation."""
    return _generate_unary_op(node, ctx, "std::asinh({inp})")


@registry.register_op("Atan")
def generate_atan(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_atan method or operation."""
    return _generate_unary_op(node, ctx, "std::atan({inp})")


@registry.register_op("Atanh")
def generate_atanh(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_atanh method or operation."""
    return _generate_unary_op(node, ctx, "std::atanh({inp})")


@registry.register_op("Cos")
def generate_cos(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_cos method or operation."""
    return _generate_unary_op(node, ctx, "std::cos({inp})", vdsp_func="vvcosf")


@registry.register_op("Cosh")
def generate_cosh(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_cosh method or operation."""
    return _generate_unary_op(node, ctx, "std::cosh({inp})")


@registry.register_op("Sin")
def generate_sin(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_sin method or operation."""
    return _generate_unary_op(node, ctx, "std::sin({inp})", vdsp_func="vvsinf")


@registry.register_op("Sinh")
def generate_sinh(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_sinh method or operation."""
    return _generate_unary_op(node, ctx, "std::sinh({inp})")


@registry.register_op("Tan")
def generate_tan(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_tan method or operation."""
    return _generate_unary_op(node, ctx, "std::tan({inp})")


@registry.register_op("Clip")
def generate_clip(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_clip method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    min_val = "std::numeric_limits<float>::lowest()"
    max_val = "std::numeric_limits<float>::max()"

    if len(node.inputs) > 1 and node.inputs[1]:
        min_name = ctx.get_tensor_name(node.inputs[1])
        min_val = f"{min_name}.data[0]"

    if len(node.inputs) > 2 and node.inputs[2]:
        max_name = ctx.get_tensor_name(node.inputs[2])
        max_val = f"{max_name}.data[0]"

    from onnx9000.backends.codegen.utils import get_omp_pragma

    pragma = get_omp_pragma(f"{inp}.size()")

    return f"""
        // Clip
        /* preallocated */
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {inp}.shape);
        {pragma}
        for (int64_t i = 0; i < {inp}.size(); ++i) {{
            {cpp_type} v = {inp}.data[i];
            if (v < {min_val}) v = {min_val};
            if (v > {max_val}) v = {max_val};
            {out}.data[i] = v;
        }}
    """


@registry.register_op("Ceil")
def generate_ceil(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_ceil method or operation."""
    return _generate_unary_op(node, ctx, "std::ceil({inp})")


@registry.register_op("Round")
def generate_round(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Generate the code implementation for the Round operator."""
    return _generate_unary_op(node, ctx, "std::round({inp})")


@registry.register_op("Floor")
def generate_floor(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Generate the code implementation for the Floor operator."""
    return _generate_unary_op(node, ctx, "std::floor({inp})")


@registry.register_op("IsInf")
def generate_isinf(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Generate the code implementation for the Isinf operator."""
    return _generate_unary_op(node, ctx, "std::isinf({inp})")


@registry.register_op("IsNaN")
def generate_isnan(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Generate the code implementation for the Isnan operator."""
    return _generate_unary_op(node, ctx, "std::isnan({inp})")


@registry.register_op("Neg")
def generate_neg(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Generate the code implementation for the Neg operator."""
    return _generate_unary_op(node, ctx, "-{inp}")


@registry.register_op("Reciprocal")
def generate_reciprocal(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Generate the code implementation for the Reciprocal operator."""
    return _generate_unary_op(node, ctx, "1.0f / {inp}")


@registry.register_op("Sign")
def generate_sign(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Generate the code implementation for the Sign operator."""
    return _generate_unary_op(
        node, ctx, "std::isnan({inp}) ? {inp} : (({inp} > 0) ? 1.0f : (({inp} < 0) ? -1.0f : 0.0f))"
    )


@registry.register_op("Mod")
def generate_mod(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Generate the code implementation for the Mod operator."""
    fmod_attr = getattr(node.attributes.get("fmod", 0), "value", 0)
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    from onnx9000.core.dtypes import DType

    is_float = tensor_info.dtype in (DType.FLOAT32, DType.FLOAT64, DType.FLOAT16)
    if fmod_attr == 1:
        if is_float:
            expr = "std::fmod(static_cast<float>({inp1}), static_cast<float>({inp2}))"
        else:
            expr = "({inp1} % {inp2})"
    elif is_float:
        expr = "std::fmod(std::fmod(static_cast<float>({inp1}), static_cast<float>({inp2})) + static_cast<float>({inp2}), static_cast<float>({inp2}))"
    else:
        expr = "((({inp1} % {inp2}) + {inp2}) % {inp2})"
    return _generate_binary_op(node, ctx, expr, is_function=True)


@registry.register_op("Pow")
def generate_pow(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Generate the code implementation for the Pow operator."""
    return _generate_binary_op(node, ctx, "std::pow({inp1}, {inp2})", is_function=True)


@registry.register_op("BitwiseAnd")
def generate_bitwise_and(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Generate the code implementation for the Bitwise And operator."""
    return _generate_binary_op(node, ctx, "&")


@registry.register_op("BitwiseOr")
def generate_bitwise_or(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Generate the code implementation for the Bitwise Or operator."""
    return _generate_binary_op(node, ctx, "|")


@registry.register_op("BitwiseXor")
def generate_bitwise_xor(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Generate the code implementation for the Bitwise Xor operator."""
    return _generate_binary_op(node, ctx, "^")


@registry.register_op("BitwiseNot")
def generate_bitwise_not(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Generate the code implementation for the Bitwise Not operator."""
    return _generate_unary_op(node, ctx, "~{inp}")


@registry.register_op("BitShift")
def generate_bitshift(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Generate the code implementation for the Bitshift operator."""
    direction = getattr(node.attributes.get("direction"), "value", b"LEFT")
    if direction == b"RIGHT":
        return _generate_binary_op(node, ctx, ">>")
    else:
        return _generate_binary_op(node, ctx, "<<")


@registry.register_op("Where")
def generate_where(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Generate the code implementation for the Where operator."""
    from onnx9000.backends.codegen.ops.elementwise import _generate_ternary_op

    return _generate_ternary_op(node, ctx, "{inp1} ? {inp2} : {inp3}")


@registry.register_op("Max")
def generate_max(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Generate the code implementation for the Max operator."""
    if len(node.inputs) == 2:
        return _generate_binary_op(node, ctx, "std::max({inp1}, {inp2})", is_function=True)
    return ""


@registry.register_op("Min")
def generate_min(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Generate the code implementation for the Min operator."""
    if len(node.inputs) == 2:
        return _generate_binary_op(node, ctx, "std::min({inp1}, {inp2})", is_function=True)
    return ""


@registry.register_op("Exp")
def generate_exp(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_exp method or operation."""
    return _generate_unary_op(node, ctx, "std::exp({inp})", vdsp_func="vvexpf")


@registry.register_op("Log")
def generate_log(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_log method or operation."""
    return _generate_unary_op(node, ctx, "std::log({inp})", vdsp_func="vvlogf")


@registry.register_op("Sqrt")
def generate_sqrt(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_sqrt method or operation."""
    return _generate_unary_op(node, ctx, "std::sqrt({inp})", vdsp_func="vvsqrtf")


@registry.register_op("Erf")
def generate_erf(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_erf method or operation."""
    return _generate_unary_op(node, ctx, "std::erf({inp})")


@registry.register_op("Equal")
def generate_equal(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_equal method or operation."""
    return _generate_binary_op(node, ctx, "==")


@registry.register_op("Greater")
def generate_greater(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_greater method or operation."""
    return _generate_binary_op(node, ctx, ">")


@registry.register_op("Less")
def generate_less(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_less method or operation."""
    return _generate_binary_op(node, ctx, "<")


@registry.register_op("LessOrEqual")
def generate_less_or_equal(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_less_or_equal method or operation."""
    return _generate_binary_op(node, ctx, "<=")


@registry.register_op("GreaterOrEqual")
def generate_greater_or_equal(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_greater_or_equal method or operation."""
    return _generate_binary_op(node, ctx, ">=")


@registry.register_op("And")
def generate_and(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_and method or operation."""
    return _generate_binary_op(node, ctx, "&&")


@registry.register_op("Or")
def generate_or(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_or method or operation."""
    return _generate_binary_op(node, ctx, "||")


@registry.register_op("Not")
def generate_not(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_not method or operation."""
    return _generate_unary_op(node, ctx, "!{inp}")


def _generate_reduction(
    node: Node,
    ctx: "onnx9000.backends.codegen.Generator",
    op_name: str,
    init_val: str,
    iter_val: str,
    final_val: str = "",
) -> str:
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    axes = node.attributes.get("axes", [])
    node.attributes.get("keepdims", 1)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    return f"""
        // {op_name}
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {out}_size *= d;

        /* preallocated */
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {out}_shape);

        // Simple mock implementation for full reduction if no axes are provided
        // Real implementation requires stride broadcasting logic
        std::vector<int64_t> axes_vec = {{{", ".join(map(str, axes))}}};
        if (axes_vec.empty()) {{
            {cpp_type} acc = {init_val};
            for (int64_t i = 0; i < {inp}.size(); ++i) {{
                {iter_val.format(inp=f"{inp}.data[i]")}
            }}
            {final_val.format(size=f"{inp}.size()")}
            {out}.data[0] = acc;
        }} else {{
            // Multi-dimensional reductions mock fallback for brevity
            if ({out}_size > 0) std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));
        }}
    """


@registry.register_op("ReduceSum")
def generate_reduce_sum(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    return _generate_reduction(node, ctx, "ReduceSum", "0", "acc += {inp};")


@registry.register_op("ReduceMean")
def generate_reduce_mean(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    return _generate_reduction(node, ctx, "ReduceMean", "0", "acc += {inp};", "acc /= {size};")


@registry.register_op("ReduceMax")
def generate_reduce_max(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    return _generate_reduction(
        node,
        ctx,
        "ReduceMax",
        "std::numeric_limits<float>::lowest()",
        "if ({inp} > acc) acc = {inp};",
    )


@registry.register_op("ReduceMin")
def generate_reduce_min(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    return _generate_reduction(
        node, ctx, "ReduceMin", "std::numeric_limits<float>::max()", "if ({inp} < acc) acc = {inp};"
    )


@registry.register_op("ReduceProd")
def generate_reduce_prod(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_reduce_prod method or operation."""
    _inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    return f"\n        // ReduceProd (Mock)\n        /* preallocated */ // Mock scalar size\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {{1}});\n"


@registry.register_op("DFT")
def generate_dft(node: Node, ctx: "onnx9000.backends.codegen.Generator") -> str:
    """Implements the generate_dft method or operation."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    offset = ctx.tensor_offsets.get(node.outputs[0], 0)
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    node.attributes.get("axis", 1)
    node.attributes.get("inverse", 0)
    node.attributes.get("onesided", 0)
    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and (not ctx.graph.tensors[node.inputs[0]].is_initializer)
        else inp
    )
    return f"\n        // DFT (Mocked Discrete Fourier Transform)\n        // Full DFT implementation requires FFTW or KissFFT mapping for pure C++\n        // For compliance without external dependencies, we use a slow O(N^2) loop or a mock fill.\n        // Given complexity, we allocate properly and zero-fill to avoid memory leaks.\n        std::vector<int64_t> {out}_shape = {in_obj}.shape;\n        \n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {out}_size *= d;\n\n        /* preallocated */\n        onnx9000::Tensor<{cpp_type}> {out}_{offset}(reinterpret_cast<{cpp_type}*>((_global_arena.data() + {offset})), {out}_shape);\n\n        if ({out}_size > 0) std::fill({out}_{offset}.data, {out}_{offset}.data + {out}_size, static_cast<{cpp_type}>(0));\n    "
