"""Keras Source Generator."""

from typing import Any

from onnx9000.core.ir import Graph, Node, Tensor


class ONNXToKerasVisitor:
    """Generates Keras 3 code from ONNX IR."""

    def __init__(self, graph: Graph):
        """Initialize the visitor."""
        self.graph = graph

    def _get_shape(self, tensor_or_name: Any) -> list[int]:
        if isinstance(tensor_or_name, Tensor):
            return list(tensor_or_name.shape) if tensor_or_name.shape else []
        elif isinstance(tensor_or_name, str) and tensor_or_name in self.graph.tensors:
            t = self.graph.tensors[tensor_or_name]
            return list(t.shape) if t.shape else []
        return []

    def generate(self) -> str:
        """Generate the Keras 3 source code."""
        lines = []
        lines.append("# type: ignore")
        lines.append("import keras")
        lines.append("from keras import layers, ops")
        lines.append("import numpy as np")
        lines.append("")

        class_name = f"Model_{self.graph.name or 'Generated'}"
        lines.append(f"class {class_name}(keras.Model):")
        lines.append("    def __init__(self, **kwargs):")
        lines.append("        super().__init__(**kwargs)")

        init_lines = []

        input_names = (
            [getattr(inp, "name", str(inp)) for inp in self.graph.inputs]
            if self.graph.inputs
            else []
        )
        if input_names:
            forward_lines = ["    def call(self, inputs):"]
            # Assume inputs passed as a list or dict if multiple
            if len(input_names) == 1:
                forward_lines.append(f"        {input_names[0]} = inputs")
            else:
                for i, name in enumerate(input_names):
                    forward_lines.append(f"        {name} = inputs[{i}]")
        else:
            forward_lines = ["    def call(self, inputs=None):"]

        env = {name: name for name in input_names}
        for init_name in self.graph.initializers:
            if init_name in env:
                del env[init_name]

        module_idx = 0
        for node in self.graph.nodes:
            out_name = node.outputs[0] if getattr(node, "outputs", None) else f"out_{module_idx}"

            inputs_mapped = []
            for inp in node.inputs:
                name = getattr(inp, "name", str(inp))
                if name in self.graph.initializers:
                    inputs_mapped.append(f"self.{name}")
                    init_line = f"        self.{name} = self.add_weight(shape={tuple(self._get_shape(name))}, initializer='zeros', trainable=False, name='{name}')"
                    if init_line not in init_lines:
                        init_lines.append(init_line)
                else:
                    inputs_mapped.append(env.get(name, name))

            attrs = getattr(node, "attributes", {}) or {}

            if node.op_type == "Conv":
                l_name = f"conv_{module_idx}"
                w_shape = self._get_shape(node.inputs[1]) if len(node.inputs) > 1 else []
                filters = w_shape[0] if w_shape else 1
                kernel_size = tuple(w_shape[2:]) if w_shape and len(w_shape) > 2 else (3, 3)
                # Keras uses data_format
                init_lines.append(
                    f"        self.{l_name} = layers.Conv2D({filters}, kernel_size={kernel_size}, data_format='channels_first')"
                )
                forward_lines.append(
                    f"        {out_name} = self.{l_name}({inputs_mapped[0] if inputs_mapped else 'None'})"
                )
            elif node.op_type == "ConvTranspose":
                l_name = f"conv_transpose_{module_idx}"
                w_shape = self._get_shape(node.inputs[1]) if len(node.inputs) > 1 else []
                filters = w_shape[1] if w_shape and len(w_shape) > 1 else 1
                kernel_size = tuple(w_shape[2:]) if w_shape and len(w_shape) > 2 else (3, 3)
                init_lines.append(
                    f"        self.{l_name} = layers.Conv2DTranspose({filters}, kernel_size={kernel_size}, data_format='channels_first')"
                )
                forward_lines.append(
                    f"        {out_name} = self.{l_name}({inputs_mapped[0] if inputs_mapped else 'None'})"
                )
            elif node.op_type in ("MatMul", "Gemm"):
                l_name = f"linear_{module_idx}"
                w_shape = self._get_shape(node.inputs[1]) if len(node.inputs) > 1 else []
                units = w_shape[1] if w_shape and len(w_shape) > 1 else 1
                init_lines.append(f"        self.{l_name} = layers.Dense({units})")
                forward_lines.append(
                    f"        {out_name} = self.{l_name}({inputs_mapped[0] if inputs_mapped else 'None'})"
                )
            elif node.op_type == "LayerNormalization":
                l_name = f"mod_{module_idx}"
                init_lines.append(f"        self.{l_name} = layers.LayerNormalization()")
                forward_lines.append(
                    f"        {out_name} = self.{l_name}({inputs_mapped[0] if inputs_mapped else 'None'})"
                )
            elif node.op_type in ("BatchNorm", "BatchNormalization"):
                l_name = f"batch_norm_{module_idx}"
                init_lines.append(f"        self.{l_name} = layers.BatchNormalization(axis=1)")
                forward_lines.append(
                    f"        {out_name} = self.{l_name}({inputs_mapped[0] if inputs_mapped else 'None'})"
                )
            elif node.op_type == "Add":
                forward_lines.append(
                    f"        {out_name} = ops.add({inputs_mapped[0] if len(inputs_mapped) > 0 else 'None'}, {inputs_mapped[1] if len(inputs_mapped) > 1 else 'None'})"
                )
            elif node.op_type == "Relu":
                forward_lines.append(
                    f"        {out_name} = ops.relu({inputs_mapped[0] if inputs_mapped else 'None'})"
                )
            elif node.op_type == "Reshape":
                forward_lines.append(
                    f"        {out_name} = ops.reshape({inputs_mapped[0]}, {inputs_mapped[1] if len(inputs_mapped) > 1 else '(-1,)'})"
                )
            elif node.op_type == "Transpose":
                perm = attrs.get("perm", [])
                forward_lines.append(
                    f"        {out_name} = ops.transpose({inputs_mapped[0]}, {tuple(perm)})"
                )
            elif node.op_type == "Shape":
                forward_lines.append(f"        {out_name} = ops.shape({inputs_mapped[0]})")
            elif node.op_type == "Gather":
                forward_lines.append(
                    f"        {out_name} = ops.take({inputs_mapped[0]}, {inputs_mapped[1]}, axis={attrs.get('axis', 0)})"
                )
            elif node.op_type == "Einsum":
                equation = attrs.get("equation", "")
                forward_lines.append(
                    f"        {out_name} = ops.einsum('{equation}', {', '.join(inputs_mapped)})"
                )
            else:
                forward_lines.append(
                    f"        {out_name} = {inputs_mapped[0] if inputs_mapped else 'None'}  # Fallback for {node.op_type}"
                )

            env[out_name] = out_name
            module_idx += 1

        if not init_lines:
            init_lines.append("        pass")

        lines.extend(init_lines)
        lines.append("")
        lines.extend(forward_lines)

        if self.graph.outputs:
            out_names = ", ".join([getattr(out, "name", str(out)) for out in self.graph.outputs])
            if len(self.graph.outputs) == 1:
                lines.append(f"        return {out_names}")
            else:
                lines.append(f"        return ({out_names})")
        else:
            lines.append("        return None")

        return "\n".join(lines)
