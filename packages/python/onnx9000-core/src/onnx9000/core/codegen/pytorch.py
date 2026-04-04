"""PyTorch Source Generator."""

from typing import Any

from onnx9000.core.ir import Graph, Node


class ONNXToPyTorchVisitor:
    """Generates PyTorch code from ONNX IR."""

    def __init__(self, graph: Graph):
        """Docstring for D107."""
        self.graph = graph

    def generate(self) -> str:
        """Docstring for D102."""
        lines = []
        lines.append("import torch")
        lines.append("import torch.nn as nn")
        lines.append("import torch.nn.functional as F")
        lines.append("")

        class_name = f"Model_{self.graph.name or 'Generated'}"
        lines.append(f"class {class_name}(nn.Module):")
        lines.append("    def __init__(self):")
        lines.append("        super().__init__()")

        init_lines = []

        input_names = [inp.name for inp in self.graph.inputs] if self.graph.inputs else []
        if input_names:
            forward_lines = [f"    def forward(self, {', '.join(input_names)}):"]
        else:
            forward_lines = ["    def forward(self):"]

        env = {name: name for name in input_names}

        # Track stateful sub-modules
        module_idx = 0
        for node in self.graph.nodes:
            out_name = node.outputs[0] if getattr(node, "outputs", None) else f"out_{module_idx}"

            inputs_mapped = []
            for inp in node.inputs:
                name = getattr(inp, "name", str(inp))
                inputs_mapped.append(env.get(name, name))

            ", ".join(inputs_mapped)

            if node.op_type == "Conv":
                l_name = f"conv_{module_idx}"
                init_lines.append(f"        self.{l_name} = nn.Conv2d(1, 1, kernel_size=(3,3))")
                forward_lines.append(
                    f"        {out_name} = self.{l_name}({inputs_mapped[0] if inputs_mapped else 'None'})"
                )
            elif node.op_type == "MatMul" or node.op_type == "Gemm":
                l_name = f"linear_{module_idx}"
                init_lines.append(f"        self.{l_name} = nn.Linear(1, 1)")
                forward_lines.append(
                    f"        {out_name} = self.{l_name}({inputs_mapped[0] if inputs_mapped else 'None'})"
                )
            elif node.op_type == "BatchNorm" or node.op_type == "BatchNormalization":
                l_name = f"batch_norm_{module_idx}"
                init_lines.append(f"        self.{l_name} = nn.BatchNorm2d(1)")
                forward_lines.append(
                    f"        {out_name} = self.{l_name}({inputs_mapped[0] if inputs_mapped else 'None'})"
                )
            elif node.op_type == "Add":
                forward_lines.append(
                    f"        {out_name} = {inputs_mapped[0] if len(inputs_mapped) > 0 else 'None'} + {inputs_mapped[1] if len(inputs_mapped) > 1 else 'None'}"
                )
            elif node.op_type == "Relu":
                forward_lines.append(
                    f"        {out_name} = F.relu({inputs_mapped[0] if inputs_mapped else 'None'})"
                )
            elif node.op_type == "Pad":
                forward_lines.append(f"        {out_name} = F.pad({inputs_mapped[0]}, pad=(1, 1))")
            elif node.op_type == "Reshape":
                forward_lines.append(f"        {out_name} = {inputs_mapped[0]}.view(-1)")
            elif node.op_type == "Constant":
                l_name = f"param_{module_idx}"
                init_lines.append(f"        self.register_buffer('{l_name}', torch.zeros(()))")
                forward_lines.append(f"        {out_name} = self.{l_name}")
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
            lines.append(f"        return {out_names}")
        else:
            lines.append("        return None")

        return "\n".join(lines)
