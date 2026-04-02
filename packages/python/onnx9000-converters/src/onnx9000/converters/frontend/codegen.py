"""Code generation for onnx9000."""

from onnx9000.converters.frontend.builder import GraphBuilder


def generate_pytorch(builder: GraphBuilder) -> str:
    """Generate PyTorch nn.Module source code from a GraphBuilder."""
    # Phase 62: Docstring Restoration
    doc_string = getattr(builder, "doc_string", "Generated PyTorch module.")
    if not doc_string:
        doc_string = "Generated PyTorch module."

    lines = [
        "import torch",
        "import torch.nn as nn",
        "",
        f"class {builder.name}(nn.Module):",
        f'    """{doc_string}"""',
        "    def __init__(self):",
        "        super().__init__()",
    ]

    # Handle parameters/weights
    if not builder.parameters:
        lines.append("        pass")
    else:
        for p in builder.parameters:
            shape = tuple(p.shape)
            lines.append(f"        self.{p.name} = nn.Parameter(torch.zeros({shape}))")

    lines.append("")
    input_names = [i.name for i in builder.inputs]
    inputs_str = ", ".join(input_names)
    lines.append(f"    def forward(self, {inputs_str}):")

    # Handle nodes
    for n in builder.nodes:
        out_names = ", ".join([o.name for o in n.outputs])
        in_names = [i.name if hasattr(i, "name") else str(i) for i in n.inputs]

        # Basic mapping back to PyTorch syntax
        if n.op_type == "Add":
            lines.append(f"        {out_names} = {in_names[0]} + {in_names[1]}")
        elif n.op_type == "Mul":
            lines.append(f"        {out_names} = {in_names[0]} * {in_names[1]}")
        elif n.op_type == "Sub":
            lines.append(f"        {out_names} = {in_names[0]} - {in_names[1]}")
        elif n.op_type == "Div":
            lines.append(f"        {out_names} = {in_names[0]} / {in_names[1]}")
        elif n.op_type == "MatMul":
            lines.append(f"        {out_names} = torch.matmul({', '.join(in_names)})")
        elif n.op_type == "Relu":
            lines.append(f"        {out_names} = torch.relu({in_names[0]})")
        else:
            # Fallback to generic torch call
            lines.append(f"        {out_names} = torch.{n.op_type.lower()}({', '.join(in_names)})")

    # Handle outputs
    output_names = [o.name for o in builder.outputs]
    if not output_names:
        lines.append("        return None")
    elif len(output_names) == 1:
        lines.append(f"        return {output_names[0]}")
    else:
        lines.append(f"        return ({', '.join(output_names)})")

    return "\n".join(lines)


def generate_keras(builder: GraphBuilder) -> str:
    """Generate Keras source code from a GraphBuilder."""
    lines = [
        "import keras",
        "from keras import layers",
        "",
        f"def {builder.name}_model():",
        '    """Generated Keras model."""',
        f"    inputs = [layers.Input(shape=tuple(i.shape[1:])) for i in {builder.inputs}]",
        "    # Keras codegen is a work in progress",
        "    return keras.Model(inputs=inputs, outputs=inputs)",
    ]
    return "\n".join(lines)


def generate_jax(builder: GraphBuilder) -> str:
    """Generate JAX source code from a GraphBuilder."""
    lines = [
        "import jax",
        "import jax.numpy as jnp",
        "",
        f"def {builder.name}_func(params, inputs):",
        '    """Generated JAX function."""',
    ]

    input_names = [i.name for i in builder.inputs]
    for i, name in enumerate(input_names):
        lines.append(f"    {name} = inputs[{i}]")

    # Handle nodes
    for n in builder.nodes:
        out_names = ", ".join([o.name for o in n.outputs])
        in_names = [i.name if hasattr(i, "name") else str(i) for i in n.inputs]

        # Basic mapping back to JAX syntax
        if n.op_type == "Add":
            lines.append(f"    {out_names} = {in_names[0]} + {in_names[1]}")
        elif n.op_type == "Mul":
            lines.append(f"    {out_names} = {in_names[0]} * {in_names[1]}")
        elif n.op_type == "Relu":
            lines.append(f"    {out_names} = jax.nn.relu({in_names[0]})")
        elif n.op_type == "MatMul":
            lines.append(f"    {out_names} = jnp.matmul({', '.join(in_names)})")
        else:
            lines.append(f"    # {n.op_type} not fully supported in JAX codegen yet")
            lines.append(f"    {out_names} = {in_names[0]}")

    # Handle outputs
    output_names = [o.name for o in builder.outputs]
    if not output_names:
        lines.append("    return None")
    elif len(output_names) == 1:
        lines.append(f"    return {output_names[0]}")
    else:
        lines.append(f"    return ({', '.join(output_names)})")

    return "\n".join(lines)
