"""
C++ Code Generation Utilities

Translates ONNX operations to equivalent C++ bindings and memory buffers.
"""

from onnx9000.backends.codegen.generator import Generator
from onnx9000.backends.codegen.utils import get_omp_pragma
from onnx9000.core.ir import Node
from onnx9000.core.registry import registry


@registry.register("ReluGrad")
def generate_relu_grad(node: Node, generator_context: Generator) -> str:
    """Provides generate relu grad functionality and verification."""

    grad_out = generator_context.get_tensor_name(node.inputs[0])
    fwd_in = generator_context.get_tensor_name(node.inputs[1])
    grad_in = generator_context.get_tensor_name(node.outputs[0])

    tensor_info = generator_context.graph.tensors[node.outputs[0]]
    buffer_idx = tensor_info.buffer_id

    cpp_type = "float"
    if tensor_info.dtype is not None:
        from onnx9000.core.dtypes import to_cpp_type

        cpp_type = to_cpp_type(tensor_info.dtype)

    pragma = get_omp_pragma(f"{fwd_in}.size()")

    return f"""
        // ReluGrad
        _arena[{buffer_idx}].resize({fwd_in}.size() * sizeof({cpp_type}));
        onnx9000::Tensor<{cpp_type}> {grad_in}(reinterpret_cast<{cpp_type}*>(_arena[{buffer_idx}].data()), {fwd_in}.shape);
        {pragma}
        for (int64_t i = 0; i < {fwd_in}.size(); ++i) {{
            {grad_in}.data[i] = ({fwd_in}.data[i] > 0.0f) ? {grad_out}.data[i] : 0.0f;
        }}
"""


@registry.register("SGDOptimizer")
def generate_sgd(node: Node, ctx: Generator) -> str:
    """Provides generate sgd functionality and verification."""
    # Inputs: param, grad, lr
    param = ctx.get_tensor_name(node.inputs[0])
    grad = ctx.get_tensor_name(node.inputs[1])
    lr = node.attributes.get("lr", 0.01)

    # Optimizer updates parameter in-place.
    # We don't allocate an output buffer.

    pragma = get_omp_pragma(f"{param}.size()")

    return f"""
        // SGDOptimizer
        {pragma}
        for (int64_t i = 0; i < {param}.size(); ++i) {{
            {param}.data[i] -= {lr}f * {grad}.data[i];
        }}
"""


@registry.register("AdamWOptimizer")
def generate_adamw(node: Node, ctx: Generator) -> str:
    """Provides generate adamw functionality and verification."""
    # Inputs: param, grad, m, v
    param = ctx.get_tensor_name(node.inputs[0])
    grad = ctx.get_tensor_name(node.inputs[1])

    # In a full graph, m and v (moments) are state tensors stored across steps.
    m = ctx.get_tensor_name(node.inputs[2])
    v = ctx.get_tensor_name(node.inputs[3])

    lr = node.attributes.get("lr", 0.001)
    beta1 = node.attributes.get("beta1", 0.9)
    beta2 = node.attributes.get("beta2", 0.999)
    eps = node.attributes.get("eps", 1e-8)
    weight_decay = node.attributes.get("weight_decay", 0.01)
    step_t = node.attributes.get(
        "step_t", 1.0
    )  # Should be a dynamic tensor but hardcoded here for mock

    pragma = get_omp_pragma(f"{param}.size()")

    return f"""
        // AdamWOptimizer
        {pragma}
        for (int64_t i = 0; i < {param}.size(); ++i) {{
            // Weight decay
            {param}.data[i] -= {lr}f * {weight_decay}f * {param}.data[i];

            // Momentum
            {m}.data[i] = {beta1}f * {m}.data[i] + (1.0f - {beta1}f) * {grad}.data[i];
            {v}.data[i] = {beta2}f * {v}.data[i] + (1.0f - {beta2}f) * {grad}.data[i] * {grad}.data[i];

            // Bias correction
            float m_hat = {m}.data[i] / (1.0f - std::pow({beta1}f, {step_t}f));
            float v_hat = {v}.data[i] / (1.0f - std::pow({beta2}f, {step_t}f));

            // Update
            {param}.data[i] -= {lr}f * m_hat / (std::sqrt(v_hat) + {eps}f);
        }}
"""
