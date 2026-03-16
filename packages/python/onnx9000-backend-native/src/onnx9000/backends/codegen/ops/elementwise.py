"""
C++ Code Generation Utilities

Translates ONNX operations to equivalent C++ bindings and memory buffers.
"""

from typing import Optional
from onnx9000.backends.codegen.generator import Generator
from onnx9000.backends.codegen.utils import get_omp_pragma
from onnx9000.core.ir import Node
from onnx9000.core.registry import global_registry as registry


def _generate_unary_op(node: Node, generator_context: Generator, op_expr: str) -> str:
    """Implements the _generate_unary_op method or operation."""
    inp = generator_context.get_tensor_name(node.inputs[0])
    out = generator_context.get_tensor_name(node.outputs[0])
    tensor_info = generator_context.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    pragma = get_omp_pragma(f"{inp}.size()")
    return f"\n        // Unary Op: {node.op_type}\n        _arena[{buffer_idx}].resize({inp}.size() * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {inp}.shape);\n        {pragma}\n        for (int64_t i = 0; i < {inp}.size(); ++i) {{\n            {out}.data[i] = {op_expr.format(inp=f'{inp}.data[i]')};\n        }}\n"


@registry.register_op("Relu")
def generate_relu(node: Node, generator_context: Generator) -> str:
    """Implements the generate_relu method or operation."""
    return _generate_unary_op(node, generator_context, "std::max(0.0f, {inp})")


@registry.register_op("Elu")
def generate_elu(node: Node, generator_context: Generator) -> str:
    """Generate the code implementation for the Elu operator."""
    alpha = node.attributes.get("alpha", 1.0)
    return _generate_unary_op(
        node,
        generator_context,
        f"({{inp}} >= 0.0f) ? {{inp}} : static_cast<float>({alpha} * (std::exp({{inp}}) - 1.0f))",
    )


@registry.register_op("Celu")
def generate_celu(node: Node, generator_context: Generator) -> str:
    """Generate the code implementation for the Celu operator."""
    alpha = node.attributes.get("alpha", 1.0)
    return _generate_unary_op(
        node,
        generator_context,
        f"std::max(0.0f, {{inp}}) + std::min(0.0f, static_cast<float>({alpha} * (std::exp({{inp}} / {alpha}) - 1.0f)))",
    )


@registry.register_op("LeakyRelu")
def generate_leaky_relu(node: Node, generator_context: Generator) -> str:
    """Generate the code implementation for the Leaky Relu operator."""
    alpha = node.attributes.get("alpha", 0.01)
    return _generate_unary_op(
        node,
        generator_context,
        f"({{inp}} >= 0.0f) ? {{inp}} : static_cast<float>({alpha} * {{inp}})",
    )


@registry.register_op("Selu")
def generate_selu(node: Node, generator_context: Generator) -> str:
    """Generate the code implementation for the Selu operator."""
    alpha = node.attributes.get("alpha", 1.6732632423543772)
    gamma = node.attributes.get("gamma", 1.0507009873554805)
    return _generate_unary_op(
        node,
        generator_context,
        f"static_cast<float>({gamma} * (({{inp}} > 0.0f) ? {{inp}} : ({alpha} * (std::exp({{inp}}) - 1.0f))))",
    )


@registry.register_op("Softplus")
def generate_softplus(node: Node, generator_context: Generator) -> str:
    """Generate the code implementation for the Softplus operator."""
    return _generate_unary_op(node, generator_context, "std::log(std::exp({inp}) + 1.0f)")


@registry.register_op("Softsign")
def generate_softsign(node: Node, generator_context: Generator) -> str:
    """Generate the code implementation for the Softsign operator."""
    return _generate_unary_op(node, generator_context, "{inp} / (1.0f + std::abs({inp}))")


@registry.register_op("ThresholdedRelu")
def generate_thresholded_relu(node: Node, generator_context: Generator) -> str:
    """Generate the code implementation for the Thresholded Relu operator."""
    alpha = node.attributes.get("alpha", 1.0)
    return _generate_unary_op(
        node, generator_context, f"({{inp}} > static_cast<float>({alpha})) ? {{inp}} : 0.0f"
    )


@registry.register_op("Mish")
def generate_mish(node: Node, generator_context: Generator) -> str:
    """Generate the code implementation for the Mish operator."""
    return _generate_unary_op(
        node, generator_context, "{inp} * std::tanh(std::log(std::exp({inp}) + 1.0f))"
    )


@registry.register_op("HardSigmoid")
def generate_hard_sigmoid(node: Node, generator_context: Generator) -> str:
    """Generate the code implementation for the Hard Sigmoid operator."""
    alpha = node.attributes.get("alpha", 0.2)
    beta = node.attributes.get("beta", 0.5)
    return _generate_unary_op(
        node,
        generator_context,
        f"std::max(0.0f, std::min(1.0f, static_cast<float>({alpha} * {{inp}} + {beta})))",
    )


@registry.register_op("HardSwish")
def generate_hard_swish(node: Node, generator_context: Generator) -> str:
    """Generate the code implementation for the Hard Swish operator."""
    return _generate_unary_op(
        node,
        generator_context,
        "{inp} * std::max(0.0f, std::min(1.0f, static_cast<float>({inp} / 6.0f + 0.5f)))",
    )


@registry.register_op("Sigmoid")
def generate_sigmoid(node: Node, generator_context: Generator) -> str:
    """Implements the generate_sigmoid method or operation."""
    return _generate_unary_op(node, generator_context, "1.0f / (1.0f + std::exp(-{inp}))")


@registry.register_op("Tanh")
def generate_tanh(node: Node, generator_context: Generator) -> str:
    """Implements the generate_tanh method or operation."""
    return _generate_unary_op(node, generator_context, "std::tanh({inp})")


def _generate_binary_op(
    node: Node,
    generator_context: Generator,
    op_expr: str,
    vdsp_func: Optional[str] = None,
    is_function: bool = False,
) -> str:
    """Implements the _generate_binary_op method or operation."""
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
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    pragma = get_omp_pragma(f"{out}.size()")

    def format_op(a, b):
        """Execute the Format op process and return the computed results."""
        if is_function:
            return op_expr.format(inp1=a, inp2=b)
        return f"{a} {op_expr} {b}"

    if in1_shape == in2_shape:
        fast_loop = f"\n        {pragma}\n        for (int64_t i = 0; i < {inp1}.size(); ++i) {{\n            {out}.data[i] = {format_op(f'{inp1}.data[i]', f'{inp2}.data[i]')};\n        }}\n"
        if vdsp_func and cpp_type == "float":
            fast_loop = f"\n#if defined(__APPLE__) && defined(USE_ACCELERATE)\n        {vdsp_func}({inp1}.data, 1, {inp2}.data, 1, {out}.data, 1, {inp1}.size());\n#else\n{fast_loop}\n#endif\n"
        return f"\n        // Binary Op: {node.op_type} (Fast Path No Broadcast)\n        _arena[{buffer_idx}].resize({inp1}.size() * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {inp1}.shape);\n{fast_loop}\n"
    out_shape_str = "{" + ", ".join(map(str, out_shape)) + "}"
    return f"\n        // Binary Op: {node.op_type} (Broadcast Path)\n        std::vector<int64_t> {out}_shape = {out_shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {out}_size *= d;\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        {pragma}\n        for (int64_t i = 0; i < {out}_size; ++i) {{\n            int64_t idx1 = onnx9000::broadcast_index(i, {out}.shape, {inp1}.shape, {inp1}.strides);\n            int64_t idx2 = onnx9000::broadcast_index(i, {out}.shape, {inp2}.shape, {inp2}.strides);\n            {out}.data[i] = {format_op(f'{inp1}.data[idx1]', f'{inp2}.data[idx2]')};\n        }}\n"


def _generate_ternary_op(node: Node, generator_context: Generator, op_expr: str) -> str:
    """Implements the _generate_ternary_op method or operation."""
    inp1 = generator_context.get_tensor_name(node.inputs[0])
    inp2 = generator_context.get_tensor_name(node.inputs[1])
    inp3 = generator_context.get_tensor_name(node.inputs[2])
    out = generator_context.get_tensor_name(node.outputs[0])
    tensor_info = generator_context.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id
    out_shape = tensor_info.shape
    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)
    pragma = get_omp_pragma(f"{out}.size()")
    in1_shape = generator_context.graph.tensors[node.inputs[0]].shape
    in2_shape = generator_context.graph.tensors[node.inputs[1]].shape
    in3_shape = generator_context.graph.tensors[node.inputs[2]].shape
    if in1_shape == in2_shape and in2_shape == in3_shape:
        fast_loop = f"\n        {pragma}\n        for (int64_t i = 0; i < {inp1}.size(); ++i) {{\n            {out}.data[i] = {op_expr.format(inp1=f'{inp1}.data[i]', inp2=f'{inp2}.data[i]', inp3=f'{inp3}.data[i]')};\n        }}\n"
        return f"\n        // Ternary Op: {node.op_type} (Fast Path No Broadcast)\n        _arena[{buffer_idx}].resize({inp1}.size() * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {inp1}.shape);\n{fast_loop}\n"
    out_shape_str = "{" + ", ".join(map(str, out_shape)) + "}"
    return f"\n        // Ternary Op: {node.op_type} (Broadcast Path)\n        std::vector<int64_t> {out}_shape = {out_shape_str};\n        int64_t {out}_size = 1;\n        for (auto d : {out}_shape) {out}_size *= d;\n\n        _arena[{buffer_idx}].resize({out}_size * sizeof({cpp_type}));\n        onnx9000::Tensor<{cpp_type}> {out}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {out}_shape);\n\n        {pragma}\n        for (int64_t i = 0; i < {out}_size; ++i) {{\n            int64_t idx1 = onnx9000::broadcast_index(i, {out}.shape, {inp1}.shape, {inp1}.strides);\n            int64_t idx2 = onnx9000::broadcast_index(i, {out}.shape, {inp2}.shape, {inp2}.strides);\n            int64_t idx3 = onnx9000::broadcast_index(i, {out}.shape, {inp3}.shape, {inp3}.strides);\n            {out}.data[i] = {op_expr.format(inp1=f'{inp1}.data[idx1]', inp2=f'{inp2}.data[idx2]', inp3=f'{inp3}.data[idx3]')};\n        }}\n"


@registry.register_op("PRelu")
def generate_prelu(node: Node, ctx: Generator) -> str:
    """Generate the code implementation for the Prelu operator."""
    return _generate_binary_op(
        node, ctx, "({inp1} < 0) ? ({inp1} * {inp2}) : {inp1}", is_function=True
    )


@registry.register_op("Add")
def generate_add(node: Node, generator_context: Generator) -> str:
    """Implements the generate_add method or operation."""
    return _generate_binary_op(node, generator_context, "+", vdsp_func="vDSP_vadd")


@registry.register_op("Sub")
def generate_sub(node: Node, generator_context: Generator) -> str:
    """Implements the generate_sub method or operation."""
    return _generate_binary_op(node, generator_context, "-")


@registry.register_op("Mul")
def generate_mul(node: Node, generator_context: Generator) -> str:
    """Implements the generate_mul method or operation."""
    return _generate_binary_op(node, generator_context, "*", vdsp_func="vDSP_vmul")


@registry.register_op("Div")
def generate_div(node: Node, generator_context: Generator) -> str:
    """Implements the generate_div method or operation."""
    return _generate_binary_op(node, generator_context, "/")
