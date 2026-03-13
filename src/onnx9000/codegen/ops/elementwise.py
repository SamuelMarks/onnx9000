"""
C++ Code Generation Utilities

Translates ONNX operations to equivalent C++ bindings and memory buffers.
"""

from typing import Optional

from onnx9000.codegen.generator import Generator
from onnx9000.codegen.utils import get_omp_pragma
from onnx9000.ir import Node
from onnx9000.registry import registry


def _generate_unary_op(node: Node, generator_context: Generator, op_expr: str) -> str:
    """_generate_unary_op docstring."""

    inp = generator_context.get_tensor_name(node.inputs[0])
    out = generator_context.get_tensor_name(node.outputs[0])

    tensor_info = generator_context.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    pragma = get_omp_pragma(f"{inp}.size()")

    return f"""
        // Unary Op: {node.op_type}
        _arena[{buffer_idx}].resize({inp}.size() * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {inp}.shape);
        {pragma}
        for (int64_t i = 0; i < {inp}.size(); ++i) {{
            {out}.data[i] = {op_expr.format(inp=f"{inp}.data[i]")};
        }}
"""


@registry.register("Relu")
def generate_relu(node: Node, generator_context: Generator) -> str:
    """generate_relu docstring."""
    return _generate_unary_op(node, generator_context, "std::max(0.0f, {inp})")


@registry.register("Elu")
def generate_elu(node: Node, generator_context: Generator) -> str:
    """Generate the code implementation for the Elu operator."""
    alpha = node.attributes.get("alpha", 1.0)
    return _generate_unary_op(
        node,
        generator_context,
        f"({{inp}} >= 0.0f) ? {{inp}} : static_cast<float>({alpha} * (std::exp({{inp}}) - 1.0f))",
    )


@registry.register("Celu")
def generate_celu(node: Node, generator_context: Generator) -> str:
    """Generate the code implementation for the Celu operator."""
    alpha = node.attributes.get("alpha", 1.0)
    return _generate_unary_op(
        node,
        generator_context,
        f"std::max(0.0f, {{inp}}) + std::min(0.0f, static_cast<float>({alpha} * (std::exp({{inp}} / {alpha}) - 1.0f)))",
    )


@registry.register("LeakyRelu")
def generate_leaky_relu(node: Node, generator_context: Generator) -> str:
    """Generate the code implementation for the Leaky Relu operator."""
    alpha = node.attributes.get("alpha", 0.01)
    return _generate_unary_op(
        node,
        generator_context,
        f"({{inp}} >= 0.0f) ? {{inp}} : static_cast<float>({alpha} * {{inp}})",
    )


@registry.register("Selu")
def generate_selu(node: Node, generator_context: Generator) -> str:
    """Generate the code implementation for the Selu operator."""
    alpha = node.attributes.get("alpha", 1.6732632423543772848170429916717)
    gamma = node.attributes.get("gamma", 1.0507009873554804934193349852946)
    return _generate_unary_op(
        node,
        generator_context,
        f"static_cast<float>({gamma} * (({{inp}} > 0.0f) ? {{inp}} : ({alpha} * (std::exp({{inp}}) - 1.0f))))",
    )


@registry.register("Softplus")
def generate_softplus(node: Node, generator_context: Generator) -> str:
    """Generate the code implementation for the Softplus operator."""
    return _generate_unary_op(
        node, generator_context, "std::log(std::exp({inp}) + 1.0f)"
    )


@registry.register("Softsign")
def generate_softsign(node: Node, generator_context: Generator) -> str:
    """Generate the code implementation for the Softsign operator."""
    return _generate_unary_op(
        node, generator_context, "{inp} / (1.0f + std::abs({inp}))"
    )


@registry.register("ThresholdedRelu")
def generate_thresholded_relu(node: Node, generator_context: Generator) -> str:
    """Generate the code implementation for the Thresholded Relu operator."""
    alpha = node.attributes.get("alpha", 1.0)
    return _generate_unary_op(
        node,
        generator_context,
        f"({{inp}} > static_cast<float>({alpha})) ? {{inp}} : 0.0f",
    )


@registry.register("Mish")
def generate_mish(node: Node, generator_context: Generator) -> str:
    """Generate the code implementation for the Mish operator."""
    return _generate_unary_op(
        node, generator_context, "{inp} * std::tanh(std::log(std::exp({inp}) + 1.0f))"
    )


@registry.register("HardSigmoid")
def generate_hard_sigmoid(node: Node, generator_context: Generator) -> str:
    """Generate the code implementation for the Hard Sigmoid operator."""
    alpha = node.attributes.get("alpha", 0.2)
    beta = node.attributes.get("beta", 0.5)
    return _generate_unary_op(
        node,
        generator_context,
        f"std::max(0.0f, std::min(1.0f, static_cast<float>({alpha} * {{inp}} + {beta})))",
    )


@registry.register("HardSwish")
def generate_hard_swish(node: Node, generator_context: Generator) -> str:
    """Generate the code implementation for the Hard Swish operator."""
    return _generate_unary_op(
        node,
        generator_context,
        "{inp} * std::max(0.0f, std::min(1.0f, static_cast<float>({inp} / 6.0f + 0.5f)))",
    )


@registry.register("Sigmoid")
def generate_sigmoid(node: Node, generator_context: Generator) -> str:
    """generate_sigmoid docstring."""

    return _generate_unary_op(
        node, generator_context, "1.0f / (1.0f + std::exp(-{inp}))"
    )


@registry.register("Tanh")
def generate_tanh(node: Node, generator_context: Generator) -> str:
    """generate_tanh docstring."""

    return _generate_unary_op(node, generator_context, "std::tanh({inp})")


def _generate_binary_op(
    node: Node,
    generator_context: Generator,
    op_expr: str,
    vdsp_func: Optional[str] = None,
    is_function: bool = False,
) -> str:
    """_generate_binary_op docstring."""

    inp1 = generator_context.get_tensor_name(node.inputs[0])
    inp2 = generator_context.get_tensor_name(node.inputs[1])
    out = generator_context.get_tensor_name(node.outputs[0])

    tensor_info = generator_context.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    in1_shape = generator_context.graph.tensors[node.inputs[0]].shape
    in2_shape = generator_context.graph.tensors[node.inputs[1]].shape
    out_shape = tensor_info.shape

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    pragma = get_omp_pragma(f"{out}.size()")

    def format_op(a, b):
        """Execute the Format op process and return the computed results."""
        if is_function:
            return op_expr.format(inp1=a, inp2=b)
        return f"{a} {op_expr} {b}"

    # Fast path: Identical shapes means no broadcasting required.
    if in1_shape == in2_shape:
        fast_loop = f"""
        {pragma}
        for (int64_t i = 0; i < {inp1}.size(); ++i) {{
            {out}.data[i] = {format_op(f"{inp1}.data[i]", f"{inp2}.data[i]")};
        }}
"""
        if vdsp_func:
            fast_loop = f"""
#if defined(__APPLE__) && defined(USE_ACCELERATE)
        if constexpr (std::is_same_v<{cpp_type}, float>) {{
            {vdsp_func}({inp1}.data, 1, {inp2}.data, 1, {out}.data, 1, {inp1}.size());
        }} else {{
{fast_loop}
        }}
#else
{fast_loop}
#endif
"""
        return f"""
        // Binary Op: {node.op_type} (Fast Path No Broadcast)
        _arena[{buffer_idx}].resize({inp1}.size() * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {inp1}.shape);
{fast_loop}
"""

    # Slow path: Dynamic broadcasting
    out_shape_str = "{" + ", ".join(map(str, out_shape)) + "}"

    return f"""
        // Binary Op: {node.op_type} (Broadcast Path)
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {out}_size *= d;

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);

        {pragma}
        for (int64_t i = 0; i < {out}_size; ++i) {{
            int64_t idx1 = onnx9000::broadcast_index(i, {out}.shape, {inp1}.shape, {inp1}.strides);
            int64_t idx2 = onnx9000::broadcast_index(i, {out}.shape, {inp2}.shape, {inp2}.strides);
            {out}.data[i] = {format_op(f"{inp1}.data[idx1]", f"{inp2}.data[idx2]")};
        }}
"""


def _generate_ternary_op(node: Node, generator_context: Generator, op_expr: str) -> str:
    """_generate_ternary_op docstring."""

    inp1 = generator_context.get_tensor_name(node.inputs[0])
    inp2 = generator_context.get_tensor_name(node.inputs[1])
    inp3 = generator_context.get_tensor_name(node.inputs[2])
    out = generator_context.get_tensor_name(node.outputs[0])

    tensor_info = generator_context.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    out_shape = tensor_info.shape

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    pragma = get_omp_pragma(f"{out}.size()")

    # Fast path: Identical shapes means no broadcasting required.
    in1_shape = generator_context.graph.tensors[node.inputs[0]].shape
    in2_shape = generator_context.graph.tensors[node.inputs[1]].shape
    in3_shape = generator_context.graph.tensors[node.inputs[2]].shape

    if in1_shape == in2_shape and in2_shape == in3_shape:
        fast_loop = f"""
        {pragma}
        for (int64_t i = 0; i < {inp1}.size(); ++i) {{
            {out}.data[i] = {op_expr.format(inp1=f"{inp1}.data[i]", inp2=f"{inp2}.data[i]", inp3=f"{inp3}.data[i]")};
        }}
"""
        return f"""
        // Ternary Op: {node.op_type} (Fast Path No Broadcast)
        _arena[{buffer_idx}].resize({inp1}.size() * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {inp1}.shape);
{fast_loop}
"""

    # Slow path: Dynamic broadcasting
    out_shape_str = "{" + ", ".join(map(str, out_shape)) + "}"

    return f"""
        // Ternary Op: {node.op_type} (Broadcast Path)
        std::vector<int64_t> {out}_shape = {out_shape_str};
        int64_t {out}_size = 1;
        for (auto d : {out}_shape) {out}_size *= d;

        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);

        {pragma}
        for (int64_t i = 0; i < {out}_size; ++i) {{
            int64_t idx1 = onnx9000::broadcast_index(i, {out}.shape, {inp1}.shape, {inp1}.strides);
            int64_t idx2 = onnx9000::broadcast_index(i, {out}.shape, {inp2}.shape, {inp2}.strides);
            int64_t idx3 = onnx9000::broadcast_index(i, {out}.shape, {inp3}.shape, {inp3}.strides);
            {out}.data[i] = {op_expr.format(inp1=f"{inp1}.data[idx1]", inp2=f"{inp2}.data[idx2]", inp3=f"{inp3}.data[idx3]")};
        }}
"""


@registry.register("PRelu")
def generate_prelu(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Prelu operator."""
    return _generate_binary_op(
        node, ctx, "({inp1} < 0) ? ({inp1} * {inp2}) : {inp1}", is_function=True
    )


@registry.register("Add")
def generate_add(node: Node, generator_context: Generator) -> str:
    """generate_add docstring."""

    return _generate_binary_op(node, generator_context, "+", vdsp_func="vDSP_vadd")


@registry.register("Sub")
def generate_sub(node: Node, generator_context: Generator) -> str:
    """generate_sub docstring."""

    return _generate_binary_op(node, generator_context, "-")


@registry.register("Mul")
def generate_mul(node: Node, generator_context: Generator) -> str:
    """generate_mul docstring."""

    return _generate_binary_op(node, generator_context, "*", vdsp_func="vDSP_vmul")


@registry.register("Div")
def generate_div(node: Node, generator_context: Generator) -> str:
    """generate_div docstring."""

    return _generate_binary_op(node, generator_context, "/")
