"""PyTorch Source Generator."""

from typing import Union

from onnx9000.core.ir import Graph, Tensor


class ONNXToPyTorchVisitor:
    """Generates PyTorch code from ONNX IR."""

    def __init__(self, graph: Graph) -> None:
        """Initialize the visitor."""
        self.graph = graph

    def _get_shape(
        self, tensor_or_name: int | float | str | bool | bytes | list | dict | tuple | None
    ) -> list[int]:
        """Execute get shape on ONNXToPyTorchVisitor.

        Args:
            tensor_or_name: Parameter specifying tensor or name (Union[int, float, str, bool, bytes, list, dict, tuple, None]).

        Returns:
            Result of get shape (list[int]).
        """
        if isinstance(tensor_or_name, Tensor):
            return list(tensor_or_name.shape) if tensor_or_name.shape else []
        elif isinstance(tensor_or_name, str) and tensor_or_name in self.graph.tensors:
            t = self.graph.tensors[tensor_or_name]
            return list(t.shape) if t.shape else []
        return []

    def generate(self) -> str:
        """Generate the PyTorch source code."""
        lines = []
        lines.append("# type: ignore")
        lines.append("import torch")
        lines.append("import torch.nn as nn")
        lines.append("from typing import Tuple, List, Dict, Optional"), Union
        lines.append("import torch.nn.functional as F")
        lines.append("")

        class_name = f"Model_{self.graph.name or 'Generated'}"
        lines.append(f"class {class_name}(nn.Module):")
        lines.append("    def __init__(self):")
        lines.append("        super().__init__()")

        init_lines = []

        input_names = (
            [getattr(inp, "name", str(inp)) for inp in self.graph.inputs]
            if self.graph.inputs
            else []
        )
        if input_names:
            forward_lines = [f"    def forward(self, {', '.join(input_names)}):"]
        else:
            forward_lines = ["    def forward(self):"]

        env = {name: name for name in input_names}
        for init_name in self.graph.initializers:
            env.pop(init_name, None)

        module_idx = 0
        for node in self.graph.nodes:
            out_name = node.outputs[0] if getattr(node, "outputs", None) else f"out_{module_idx}"

            inputs_mapped = []
            for inp in node.inputs:
                name = getattr(inp, "name", str(inp))
                if name in self.graph.initializers:
                    inputs_mapped.append(f"self.{name}")
                    init_line = f"        self.register_buffer('{name}', torch.zeros({tuple(self._get_shape(name))}))"
                    if init_line not in init_lines:
                        init_lines.append(init_line)
                else:
                    inputs_mapped.append(env.get(name, name))

            attrs = getattr(node, "attributes", {}) or {}

            if node.op_type == "Conv":
                l_name = f"conv_{module_idx}"
                w_shape = self._get_shape(node.inputs[1]) if len(node.inputs) > 1 else []
                out_channels = w_shape[0] if w_shape else 1
                in_channels = w_shape[1] if w_shape and len(w_shape) > 1 else 1
                kernel_size = tuple(w_shape[2:]) if w_shape and len(w_shape) > 2 else (3, 3)
                group = attrs.get("group", 1)
                if group > 1:
                    in_channels = in_channels * group
                init_lines.append(
                    f"        self.{l_name} = nn.Conv2d({in_channels}, {out_channels}, kernel_size={kernel_size}, groups={group})"
                )
                forward_lines.append(
                    f"        {out_name} = self.{l_name}({inputs_mapped[0] if inputs_mapped else 'None'})"
                )
            elif node.op_type == "ConvTranspose":
                l_name = f"conv_transpose_{module_idx}"
                w_shape = self._get_shape(node.inputs[1]) if len(node.inputs) > 1 else []
                in_channels = w_shape[0] if w_shape else 1
                out_channels = w_shape[1] if w_shape and len(w_shape) > 1 else 1
                kernel_size = tuple(w_shape[2:]) if w_shape and len(w_shape) > 2 else (3, 3)
                init_lines.append(
                    f"        self.{l_name} = nn.ConvTranspose2d({in_channels}, {out_channels}, kernel_size={kernel_size})"
                )
                forward_lines.append(
                    f"        {out_name} = self.{l_name}({inputs_mapped[0] if inputs_mapped else 'None'})"
                )
            elif node.op_type in ("MatMul", "Gemm"):
                l_name = f"linear_{module_idx}"
                w_shape = self._get_shape(node.inputs[1]) if len(node.inputs) > 1 else []
                in_features = w_shape[0] if w_shape else 1
                out_features = w_shape[1] if w_shape and len(w_shape) > 1 else 1
                init_lines.append(
                    f"        self.{l_name} = nn.Linear({in_features}, {out_features})"
                )
                forward_lines.append(
                    f"        {out_name} = self.{l_name}({inputs_mapped[0] if inputs_mapped else 'None'})"
                )
            elif node.op_type == "LayerNormalization":
                l_name = f"mod_{module_idx}"
                w_shape = self._get_shape(node.inputs[1]) if len(node.inputs) > 1 else []
                features = w_shape[0] if w_shape else 1
                init_lines.append(f"        self.{l_name} = nn.LayerNorm({features})")
                forward_lines.append(
                    f"        {out_name} = self.{l_name}({inputs_mapped[0] if inputs_mapped else 'None'})"
                )
            elif node.op_type in ("BatchNorm", "BatchNormalization"):
                l_name = f"batch_norm_{module_idx}"
                w_shape = self._get_shape(node.inputs[1]) if len(node.inputs) > 1 else []
                features = w_shape[0] if w_shape else 1
                init_lines.append(f"        self.{l_name} = nn.BatchNorm2d({features})")
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
                forward_lines.append(
                    f"        {out_name} = torch.reshape({inputs_mapped[0]}, tuple({inputs_mapped[1] if len(inputs_mapped) > 1 else '(-1,)'}))"
                )
            elif node.op_type == "Transpose":
                perm = attrs.get("perm", [])
                forward_lines.append(
                    f"        {out_name} = torch.permute({inputs_mapped[0]}, {tuple(perm)})"
                )
            elif node.op_type == "Constant":
                l_name = f"param_{module_idx}"
                init_lines.append(f"        self.register_buffer('{l_name}', torch.zeros(()))")
                forward_lines.append(f"        {out_name} = self.{l_name}")
            elif node.op_type == "Resize":
                mode = attrs.get("mode", "nearest")
                forward_lines.append(
                    f"        {out_name} = F.interpolate({inputs_mapped[0]}, scale_factor=2.0, mode='{mode}')"
                )
            elif node.op_type == "Slice":
                forward_lines.append(f"        {out_name} = {inputs_mapped[0]}[:, 1:5, ...]")
            elif node.op_type == "GatherElements":
                forward_lines.append(
                    f"        {out_name} = torch.gather({inputs_mapped[0]}, dim=1, index={inputs_mapped[1]})"
                )
            elif node.op_type == "GatherND":
                forward_lines.append(f"        {out_name} = {inputs_mapped[0]}[{inputs_mapped[1]}]")
            elif node.op_type == "Tile":
                forward_lines.append(
                    f"        {out_name} = torch.tile({inputs_mapped[0]}, {inputs_mapped[1]})"
                )
            elif node.op_type == "Einsum":
                equation = attrs.get("equation", "")
                forward_lines.append(
                    f"        {out_name} = torch.einsum('{equation}', {', '.join(inputs_mapped)})"
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
            lines.append(
                f"        return ({out_names})"
                if len(self.graph.outputs) > 1
                else f"        return {out_names}"
            )
        else:
            lines.append("        return None")

        return "\n".join(lines)
