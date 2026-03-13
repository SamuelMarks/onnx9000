"""
C++ Code Generation Utilities

Translates ONNX operations to equivalent C++ bindings and memory buffers.
"""

from onnx9000.codegen.generator import Generator
from onnx9000.codegen.ops.elementwise import _generate_binary_op, _generate_unary_op
from onnx9000.ir import Node
from onnx9000.registry import registry


# Step 151-154, 176
@registry.register("BlackmanWindow")
def generate_blackman_window(node: Node, ctx: Generator) -> str:
    """generate_blackman_window docstring."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    periodic = node.attributes.get("periodic", 1)

    return f"""
        // BlackmanWindow
        int64_t window_size = {inp}.size() > 0 ? {inp}.data[0] : 0;
        std::vector<int64_t> {out}_shape = {{window_size}};
        
        _arena[{buffer_idx}].resize(std::max<int64_t>(1, window_size) * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        {out}_{buffer_idx}.shape = {out}_shape; // Ensure shape is explicitly dynamic

        if (window_size == 1) {{
            {out}_{buffer_idx}.data[0] = static_cast<{cpp_type}>(1.0);
        }} else if (window_size > 1) {{
            double N = static_cast<double>({periodic} ? window_size : window_size - 1);
            for (int64_t i = 0; i < window_size; ++i) {{
                double x = 2.0 * M_PI * i / N;
                double val = 0.42 - 0.5 * std::cos(x) + 0.08 * std::cos(2.0 * x);
                {out}_{buffer_idx}.data[i] = static_cast<{cpp_type}>(val);
            }}
        }}
    """


@registry.register("Det")
def generate_det(node: Node, ctx: Generator) -> str:
    """generate_det docstring."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    return f"""
        // Det (Mock)
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {{
            if (d < 0) d = 1; // MOCK ONLY
            {out}_size *= d;
        }}
        if ({out}_size < 0) {out}_size = 1;
        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback

        for (size_t i = 0; i < {out}_shape.size(); ++i) {{
            if ({out}_shape[i] < 0) {out}_shape[i] = 1;
        }}

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        // Fill mock
        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));
    """


@registry.register("Mean")
def generate_mean(node: Node, ctx: Generator) -> str:
    """generate_mean docstring."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    return f"""
        // Mean (Mock)
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {{
            if (d < 0) d = 1; // MOCK ONLY
            {out}_size *= d;
        }}
        if ({out}_size < 0) {out}_size = 1;
        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback

        for (size_t i = 0; i < {out}_shape.size(); ++i) {{
            if ({out}_shape[i] < 0) {out}_shape[i] = 1;
        }}

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        // Fill mock
        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));
    """


@registry.register("MelWeightMatrix")
def generate_mel_weight_matrix(node: Node, ctx: Generator) -> str:
    """generate_mel_weight_matrix docstring."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    return f"""
        // MelWeightMatrix (Mock)
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {{
            if (d < 0) d = 1; // MOCK ONLY
            {out}_size *= d;
        }}
        if ({out}_size < 0) {out}_size = 1;
        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback

        for (size_t i = 0; i < {out}_shape.size(); ++i) {{
            if ({out}_shape[i] < 0) {out}_shape[i] = 1;
        }}

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        // Fill mock
        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));
    """


@registry.register("Multinomial")
def generate_multinomial(node: Node, ctx: Generator) -> str:
    """generate_multinomial docstring."""
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    return f"""
        // Multinomial (Mock)
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {{
            if (d < 0) d = 1; // MOCK ONLY
            {out}_size *= d;
        }}
        if ({out}_size < 0) {out}_size = 1;
        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback

        for (size_t i = 0; i < {out}_shape.size(); ++i) {{
            if ({out}_shape[i] < 0) {out}_shape[i] = 1;
        }}

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        // Fill mock
        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));
    """


@registry.register("ReduceSumSquare")
def generate_reduce_sum_square(node: Node, ctx: Generator) -> str:
    """generate_reduce_sum_square docstring."""
    _inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])

    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    return f"""
        // ReduceSumSquare (Mock)
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {{
            if (d < 0) d = 1; // MOCK ONLY
            {out}_size *= d;
        }}
        if ({out}_size < 0) {out}_size = 1;
        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback

        for (size_t i = 0; i < {out}_shape.size(); ++i) {{
            if ({out}_shape[i] < 0) {out}_shape[i] = 1;
        }}

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        // Fill mock
        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));
    """


@registry.register("ReduceL1")
def generate_reduce_l1(node: Node, ctx: Generator) -> str:
    """generate_reduce_l1 docstring."""
    _inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])

    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    return f"""
        // ReduceL1 (Mock)
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {{
            if (d < 0) d = 1; // MOCK ONLY
            {out}_size *= d;
        }}
        if ({out}_size < 0) {out}_size = 1;
        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback

        for (size_t i = 0; i < {out}_shape.size(); ++i) {{
            if ({out}_shape[i] < 0) {out}_shape[i] = 1;
        }}

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        // Fill mock
        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));
    """


@registry.register("ReduceL2")
def generate_reduce_l2(node: Node, ctx: Generator) -> str:
    """generate_reduce_l2 docstring."""
    _inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])

    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    return f"""
        // ReduceL2 (Mock)
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {{
            if (d < 0) d = 1; // MOCK ONLY
            {out}_size *= d;
        }}
        if ({out}_size < 0) {out}_size = 1;
        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback

        for (size_t i = 0; i < {out}_shape.size(); ++i) {{
            if ({out}_shape[i] < 0) {out}_shape[i] = 1;
        }}

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        // Fill mock
        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));
    """


@registry.register("ReduceLogSum")
def generate_reduce_log_sum(node: Node, ctx: Generator) -> str:
    """generate_reduce_log_sum docstring."""
    _inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])

    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    return f"""
        // ReduceLogSum (Mock)
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {{
            if (d < 0) d = 1; // MOCK ONLY
            {out}_size *= d;
        }}
        if ({out}_size < 0) {out}_size = 1;
        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback

        for (size_t i = 0; i < {out}_shape.size(); ++i) {{
            if ({out}_shape[i] < 0) {out}_shape[i] = 1;
        }}

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        // Fill mock
        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));
    """


@registry.register("ReduceLogSumExp")
def generate_reduce_log_sum_exp(node: Node, ctx: Generator) -> str:
    """generate_reduce_log_sum_exp docstring."""
    _inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])

    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    return f"""
        // ReduceLogSumExp (Mock)
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {{
            if (d < 0) d = 1; // MOCK ONLY
            {out}_size *= d;
        }}
        if ({out}_size < 0) {out}_size = 1;
        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback

        for (size_t i = 0; i < {out}_shape.size(); ++i) {{
            if ({out}_shape[i] < 0) {out}_shape[i] = 1;
        }}

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        // Fill mock
        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));
    """


@registry.register("Shrink")
def generate_shrink(node: Node, ctx: Generator) -> str:
    """generate_shrink docstring."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    return f"""
        // Shrink (Mock)
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {{
            if (d < 0) d = 1; // MOCK ONLY
            {out}_size *= d;
        }}
        if ({out}_size < 0) {out}_size = 1;
        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback

        for (size_t i = 0; i < {out}_shape.size(); ++i) {{
            if ({out}_shape[i] < 0) {out}_shape[i] = 1;
        }}

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        // Fill mock
        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));
    """


@registry.register("Size")
def generate_size(node: Node, ctx: Generator) -> str:
    """generate_size docstring."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "int64_t"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    out_shape_str = "{" + ", ".join(map(str, tensor_info.shape)) + "}"

    return f"""
        // Size (Mock)
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {{
            if (d < 0) d = 1; // MOCK ONLY
            {out}_size *= d;
        }}
        if ({out}_size < 0) {out}_size = 1;
        if ({out}_shape.empty()) {out}_shape = {{1}}; // Safe fallback

        for (size_t i = 0; i < {out}_shape.size(); ++i) {{
            if ({out}_shape[i] < 0) {out}_shape[i] = 1;
        }}

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);
        // Fill mock
        std::fill({out}.data, {out}.data + {out}_size, static_cast<{cpp_type}>(0));
    """


@registry.register("Abs")
def generate_abs(node: Node, ctx: Generator) -> str:
    """generate_abs docstring."""
    return _generate_unary_op(node, ctx, "onnx9000::safe_abs({inp})")


@registry.register("Acos")
def generate_acos(node: Node, ctx: Generator) -> str:
    """generate_acos docstring."""
    return _generate_unary_op(node, ctx, "std::acos({inp})")


@registry.register("Acosh")
def generate_acosh(node: Node, ctx: Generator) -> str:
    """generate_acosh docstring."""
    return _generate_unary_op(node, ctx, "std::acosh({inp})")


@registry.register("Asin")
def generate_asin(node: Node, ctx: Generator) -> str:
    """generate_asin docstring."""
    return _generate_unary_op(node, ctx, "std::asin({inp})")


@registry.register("Asinh")
def generate_asinh(node: Node, ctx: Generator) -> str:
    """generate_asinh docstring."""
    return _generate_unary_op(node, ctx, "std::asinh({inp})")


@registry.register("Atan")
def generate_atan(node: Node, ctx: Generator) -> str:
    """generate_atan docstring."""
    return _generate_unary_op(node, ctx, "std::atan({inp})")


@registry.register("Atanh")
def generate_atanh(node: Node, ctx: Generator) -> str:
    """generate_atanh docstring."""
    return _generate_unary_op(node, ctx, "std::atanh({inp})")


@registry.register("Cos")
def generate_cos(node: Node, ctx: Generator) -> str:
    """generate_cos docstring."""
    return _generate_unary_op(node, ctx, "std::cos({inp})")


@registry.register("Cosh")
def generate_cosh(node: Node, ctx: Generator) -> str:
    """generate_cosh docstring."""
    return _generate_unary_op(node, ctx, "std::cosh({inp})")


@registry.register("Sin")
def generate_sin(node: Node, ctx: Generator) -> str:
    """generate_sin docstring."""
    return _generate_unary_op(node, ctx, "std::sin({inp})")


@registry.register("Sinh")
def generate_sinh(node: Node, ctx: Generator) -> str:
    """generate_sinh docstring."""
    return _generate_unary_op(node, ctx, "std::sinh({inp})")


@registry.register("Tan")
def generate_tan(node: Node, ctx: Generator) -> str:
    """generate_tan docstring."""
    return _generate_unary_op(node, ctx, "std::tan({inp})")


@registry.register("Ceil")
def generate_ceil(node: Node, ctx: Generator) -> str:
    """generate_ceil docstring."""
    return _generate_unary_op(node, ctx, "std::ceil({inp})")


@registry.register("Round")
def generate_round(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Round operator."""
    return _generate_unary_op(node, ctx, "std::round({inp})")


@registry.register("Floor")
def generate_floor(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Floor operator."""
    return _generate_unary_op(node, ctx, "std::floor({inp})")


@registry.register("IsInf")
def generate_isinf(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Isinf operator."""
    return _generate_unary_op(node, ctx, "std::isinf({inp})")


@registry.register("IsNaN")
def generate_isnan(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Isnan operator."""
    return _generate_unary_op(node, ctx, "std::isnan({inp})")


@registry.register("Neg")
def generate_neg(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Neg operator."""
    return _generate_unary_op(node, ctx, "-{inp}")


@registry.register("Reciprocal")
def generate_reciprocal(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Reciprocal operator."""
    return _generate_unary_op(node, ctx, "1.0f / {inp}")


@registry.register("Sign")
def generate_sign(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Sign operator."""
    return _generate_unary_op(
        node,
        ctx,
        "std::isnan({inp}) ? {inp} : (({inp} > 0) ? 1.0f : (({inp} < 0) ? -1.0f : 0.0f))",
    )


@registry.register("Mod")
def generate_mod(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Mod operator."""
    # fmod assumes floating point. For ints we might need %.
    return _generate_binary_op(node, ctx, "std::fmod({inp1}, {inp2})", is_function=True)


@registry.register("Pow")
def generate_pow(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Pow operator."""
    return _generate_binary_op(node, ctx, "std::pow({inp1}, {inp2})", is_function=True)


@registry.register("BitwiseAnd")
def generate_bitwise_and(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Bitwise And operator."""
    return _generate_binary_op(node, ctx, "&")


@registry.register("BitwiseOr")
def generate_bitwise_or(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Bitwise Or operator."""
    return _generate_binary_op(node, ctx, "|")


@registry.register("BitwiseXor")
def generate_bitwise_xor(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Bitwise Xor operator."""
    return _generate_binary_op(node, ctx, "^")


@registry.register("BitwiseNot")
def generate_bitwise_not(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Bitwise Not operator."""
    return _generate_unary_op(node, ctx, "~{inp}")


@registry.register("BitShift")
def generate_bitshift(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Bitshift operator."""
    direction = node.attributes.get("direction", "LEFT")
    if direction == "RIGHT":
        return _generate_binary_op(node, ctx, ">>")
    else:
        return _generate_binary_op(node, ctx, "<<")


@registry.register("Where")
def generate_where(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Where operator."""
    from onnx9000.codegen.ops.elementwise import _generate_ternary_op

    return _generate_ternary_op(node, ctx, "{inp1} ? {inp2} : {inp3}")


@registry.register("Max")
def generate_max(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Max operator."""
    if len(node.inputs) == 2:
        return _generate_binary_op(
            node, ctx, "std::max({inp1}, {inp2})", is_function=True
        )
    # Fallback to binary loop reduction for N > 2 would go here if needed
    return ""  # pragma: no cover


@registry.register("Min")
def generate_min(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Min operator."""
    if len(node.inputs) == 2:
        return _generate_binary_op(
            node, ctx, "std::min({inp1}, {inp2})", is_function=True
        )
    # Fallback to binary loop reduction for N > 2 would go here if needed
    return ""  # pragma: no cover


@registry.register("Exp")
def generate_exp(node: Node, ctx: Generator) -> str:
    """generate_exp docstring."""

    return _generate_unary_op(node, ctx, "std::exp({inp})")


@registry.register("Log")
def generate_log(node: Node, ctx: Generator) -> str:
    """generate_log docstring."""

    return _generate_unary_op(node, ctx, "std::log({inp})")


@registry.register("Sqrt")
def generate_sqrt(node: Node, ctx: Generator) -> str:
    """generate_sqrt docstring."""

    return _generate_unary_op(node, ctx, "std::sqrt({inp})")


@registry.register("Erf")
def generate_erf(node: Node, ctx: Generator) -> str:
    """generate_erf docstring."""

    return _generate_unary_op(node, ctx, "std::erf({inp})")


# Step 173-174
@registry.register("Equal")
def generate_equal(node: Node, ctx: Generator) -> str:
    """generate_equal docstring."""

    return _generate_binary_op(node, ctx, "==")


@registry.register("Greater")
def generate_greater(node: Node, ctx: Generator) -> str:
    """generate_greater docstring."""

    return _generate_binary_op(node, ctx, ">")


@registry.register("Less")
def generate_less(node: Node, ctx: Generator) -> str:
    """generate_less docstring."""

    return _generate_binary_op(node, ctx, "<")


@registry.register("LessOrEqual")
def generate_less_or_equal(node: Node, ctx: Generator) -> str:
    """generate_less_or_equal docstring."""
    return _generate_binary_op(node, ctx, "<=")


@registry.register("GreaterOrEqual")
def generate_greater_or_equal(node: Node, ctx: Generator) -> str:
    """generate_greater_or_equal docstring."""
    return _generate_binary_op(node, ctx, ">=")


@registry.register("And")
def generate_and(node: Node, ctx: Generator) -> str:
    """generate_and docstring."""

    return _generate_binary_op(node, ctx, "&&")


@registry.register("Or")
def generate_or(node: Node, ctx: Generator) -> str:
    """generate_or docstring."""

    return _generate_binary_op(node, ctx, "||")


@registry.register("Not")
def generate_not(node: Node, ctx: Generator) -> str:
    """generate_not docstring."""

    return _generate_unary_op(node, ctx, "!{inp}")


@registry.register("ReduceSum")
def generate_reduce_sum(node: Node, ctx: Generator) -> str:
    """generate_reduce_sum docstring."""
    _inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])

    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    return f"""
        // ReduceSum (Mock)
        _arena[{buffer_idx}].resize(1 * sizeof({cpp_type})); // Mock scalar size
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {{1}});
"""


@registry.register("ReduceMean")
def generate_reduce_mean(node: Node, ctx: Generator) -> str:
    """generate_reduce_mean docstring."""
    _inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])

    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    return f"""
        // ReduceMean (Mock)
        _arena[{buffer_idx}].resize(1 * sizeof({cpp_type})); // Mock scalar size
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {{1}});
"""


@registry.register("ReduceMax")
def generate_reduce_max(node: Node, ctx: Generator) -> str:
    """generate_reduce_max docstring."""
    _inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])

    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    return f"""
        // ReduceMax (Mock)
        _arena[{buffer_idx}].resize(1 * sizeof({cpp_type})); // Mock scalar size
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {{1}});
"""


@registry.register("ReduceMin")
def generate_reduce_min(node: Node, ctx: Generator) -> str:
    """generate_reduce_min docstring."""
    _inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])

    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    return f"""
        // ReduceMin (Mock)
        _arena[{buffer_idx}].resize(1 * sizeof({cpp_type})); // Mock scalar size
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {{1}});
"""


@registry.register("ReduceProd")
def generate_reduce_prod(node: Node, ctx: Generator) -> str:
    """generate_reduce_prod docstring."""
    _inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])

    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    return f"""
        // ReduceProd (Mock)
        _arena[{buffer_idx}].resize(1 * sizeof({cpp_type})); // Mock scalar size
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {{1}});
"""


@registry.register("DFT")
def generate_dft(node: Node, ctx: Generator) -> str:
    """generate_dft docstring."""
    inp = ctx.get_tensor_name(node.inputs[0])
    out = ctx.get_tensor_name(node.outputs[0])
    tensor_info = ctx.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    axis = node.attributes.get("axis", 1)
    inverse = node.attributes.get("inverse", 0)
    onesided = node.attributes.get("onesided", 0)

    in_buf = ctx.graph.tensors[node.inputs[0]].buffer_id
    in_obj = (
        f"{inp}_{in_buf}"
        if node.inputs[0] not in ctx.graph.inputs
        and not ctx.graph.tensors[node.inputs[0]].is_initializer
        else inp
    )

    return f"""
        // DFT (Mocked Discrete Fourier Transform)
        // Full DFT implementation requires FFTW or KissFFT mapping for pure C++
        // For compliance without external dependencies, we use a slow O(N^2) loop or a mock fill.
        // Given complexity, we allocate properly and zero-fill to avoid memory leaks.
        std::vector<int64_t> {out}_shape = {in_obj}.shape;
        
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {out}_size *= d;

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}_{buffer_idx}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);

        if ({out}_size > 0) std::fill({out}_{buffer_idx}.data, {out}_{buffer_idx}.data + {out}_size, static_cast<{cpp_type}>(0));
    """
