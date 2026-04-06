"""Flax NNX Source Generator."""

from typing import Any

from onnx9000.core.ir import Graph, Node


class ONNXToFlaxNNXVisitor:
    """Generates Flax NNX code from ONNX IR."""

    def __init__(self, graph: Graph):
        """Docstring for D107."""
        self.graph = graph

    def generate(self) -> str:
        """Docstring for D102."""
        lines = []
        lines.append("# type: ignore")
        lines.append("import jax")
        lines.append("import jax.numpy as jnp")
        lines.append("import flax.linen as nn")
        lines.append("from typing import Any, Tuple, List, Dict, Optional")
        lines.append("import jax.numpy as jnp")
        lines.append("import flax.nnx as nnx")
        lines.append("")

        # load_weights helper function
        lines.append("def load_weights(nnx_module, state_dict):")
        lines.append("    # Map state dict to Flax nnx module parameters")
        lines.append("    for name, value in state_dict.items():")
        lines.append("        if hasattr(nnx_module, name):")
        lines.append("            setattr(nnx_module, name, value)")
        lines.append("")

        class_name = f"Model_{self.graph.name or 'Generated'}"
        lines.append(f"class {class_name}(nnx.Module):")
        lines.append("    def __init__(self, rngs: nnx.Rngs):")

        init_lines = []

        input_names = [inp.name for inp in self.graph.inputs] if self.graph.inputs else []
        if input_names:
            forward_lines = [f"    def __call__(self, {', '.join(input_names)}):"]
        else:
            forward_lines = ["    def __call__(self):"]

        env = {name: name for name in input_names}

        # Track stateful sub-modules
        module_idx = 0
        for node in self.graph.nodes:
            out_names = (
                [out.name if hasattr(out, "name") else str(out) for out in node.outputs]
                if getattr(node, "outputs", None)
                else [f"out_{module_idx}"]
            )
            out_name = out_names[0] if len(out_names) == 1 else ", ".join(out_names)

            inputs_mapped = []
            for inp in node.inputs:
                name = getattr(inp, "name", str(inp))
                inputs_mapped.append(env.get(name, name))

            # Handle attributes
            attrs = getattr(node, "attributes", {}) or {}

            if node.op_type == "Conv":
                l_name = f"conv_{module_idx}"
                init_lines.append(
                    f"        self.{l_name} = nnx.Conv(1, 1, kernel_size=(3,3), rngs=rngs)"
                )
                forward_lines.append(
                    f"        {out_name} = self.{l_name}({inputs_mapped[0] if inputs_mapped else 'None'})"
                )
            elif node.op_type == "ConvTranspose":
                l_name = f"conv_transpose_{module_idx}"
                init_lines.append(
                    f"        self.{l_name} = nnx.ConvTranspose(1, 1, kernel_size=(3,3), rngs=rngs)"
                )
                forward_lines.append(
                    f"        {out_name} = self.{l_name}({inputs_mapped[0] if inputs_mapped else 'None'})"
                )
            elif node.op_type == "MatMul" or node.op_type == "Gemm":
                l_name = f"linear_{module_idx}"
                init_lines.append(f"        self.{l_name} = nnx.Linear(1, 1, rngs=rngs)")
                forward_lines.append(
                    f"        {out_name} = self.{l_name}({inputs_mapped[0] if inputs_mapped else 'None'})"
                )
            elif node.op_type == "BatchNorm" or node.op_type == "BatchNormalization":
                l_name = f"batch_norm_{module_idx}"
                init_lines.append(f"        self.{l_name} = nnx.BatchNorm(1, rngs=rngs)")
                forward_lines.append(
                    f"        {out_name} = self.{l_name}({inputs_mapped[0]}, use_running_average=not self.is_training())"
                )
            elif node.op_type == "MultiHeadAttention":
                l_name = f"mha_{module_idx}"
                init_lines.append(
                    f"        self.{l_name} = nnx.MultiHeadAttention(num_heads=8, in_features=1, rngs=rngs)"
                )
                # Assuming query, key, value ordering
                qkv_args = ", ".join(inputs_mapped[:3])
                forward_lines.append(f"        {out_name} = self.{l_name}({qkv_args})")
            elif node.op_type == "Add":
                forward_lines.append(
                    f"        {out_name} = {inputs_mapped[0]} + {inputs_mapped[1]}"
                )
            elif node.op_type == "Relu":
                forward_lines.append(f"        {out_name} = jax.nn.relu({inputs_mapped[0]})")
            elif node.op_type == "Pad":
                mode = attrs.get("mode", "constant")
                pad_width = inputs_mapped[1] if len(inputs_mapped) > 1 else "((0,0),)"
                forward_lines.append(
                    f"        {out_name} = jnp.pad({inputs_mapped[0]}, pad_width={pad_width}, mode='{mode}')"
                )
            elif node.op_type == "Split":
                split_axis = attrs.get("axis", 0)
                forward_lines.append(
                    f"        {out_name} = jnp.split({inputs_mapped[0]}, len({len(out_names)}), axis={split_axis})"
                )
            elif node.op_type == "Einsum":
                equation = attrs.get("equation", "")
                forward_lines.append(
                    f"        {out_name} = jnp.einsum('{equation}', {', '.join(inputs_mapped)})"
                )
            elif node.op_type == "Softmax":
                axis = attrs.get("axis", -1)
                forward_lines.append(
                    f"        {out_name} = jax.nn.softmax({inputs_mapped[0]}, axis={axis})"
                )
            elif node.op_type == "RandomNormal":
                forward_lines.append(
                    f"        {out_name} = jax.random.normal(rngs.next(), shape=())"
                )
            elif node.op_type == "Dropout":
                l_name = f"dropout_{module_idx}"
                init_lines.append(f"        self.{l_name} = nnx.Dropout(rate=0.5, rngs=rngs)")
                forward_lines.append(
                    f"        {out_name} = self.{l_name}({inputs_mapped[0]}, deterministic=self.is_training() == False)"
                )
            elif node.op_type == "RNN":
                l_name = f"rnn_state_{module_idx}"
                init_lines.append(f"        self.{l_name} = nnx.Variable(jnp.zeros((1, 1)))")
                forward_lines.append(
                    f"        {out_name} = {inputs_mapped[0]} + self.{l_name}.value"
                )
            else:
                forward_lines.append(
                    f"        {out_name} = {inputs_mapped[0] if inputs_mapped else 'None'}  # Fallback for {node.op_type}"
                )

            for o in out_names:
                env[o] = o
            module_idx += 1

        if not init_lines:
            init_lines.append("        pass")

        lines.extend(init_lines)
        lines.append("")
        lines.extend(forward_lines)

        if self.graph.outputs:
            out_names = ", ".join([getattr(out, "name", str(out)) for out in self.graph.outputs])
            lines.append(f"        return {out_names}")
        else:
            lines.append("        return None")

        return "\n".join(lines)
