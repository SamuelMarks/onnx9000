"""FX IR parser for onnx9000."""

import torch
import torch.fx
from onnx9000.converters.frontend.builder import GraphBuilder
from onnx9000.converters.frontend.tensor import Node, Tensor
from onnx9000.core.dtypes import DType


class FXParser:
    """Parser for torch.fx.GraphModule."""

    def __init__(self, gm: torch.fx.GraphModule) -> None:
        """Initialize the FX parser."""
        self.gm = gm
        self.graph = gm.graph
        self.builder = GraphBuilder(name=getattr(gm, "__class__", "FXGraph").__name__)
        self.node_map = {}

    def _get_dtype(self, torch_dtype: torch.dtype) -> DType:
        """Map Torch dtype to DType."""
        if torch_dtype == torch.float32:
            return DType.FLOAT32
        if torch_dtype == torch.int64:
            return DType.INT64
        return DType.FLOAT32

    def _get_shape(self, shape_tuple) -> tuple:
        """Handle symbolic shapes (SymInt) from torch.export."""
        if shape_tuple is None:
            return ()
        out_shape = []
        for d in shape_tuple:
            if isinstance(d, int):
                out_shape.append(d)
            else:
                out_shape.append(str(d))
        return tuple(out_shape)

    def parse(self) -> GraphBuilder:
        """Parse the FX graph into a GraphBuilder."""
        for node in self.graph.nodes:
            self._parse_node(node)
        return self.builder

    def _parse_node(self, node: torch.fx.Node) -> None:
        """Parse a single FX node."""
        if node.op == "placeholder":
            # Input
            shape_info = node.meta.get("tensor_meta", None)
            shape = self._get_shape(shape_info.shape) if shape_info else ()
            dtype = self._get_dtype(shape_info.dtype) if shape_info else DType.FLOAT32
            tensor = Tensor(shape=shape, dtype=dtype, name=node.name)
            self.builder.inputs.append(tensor)
            self.node_map[node] = tensor

        elif node.op == "get_attr":
            # Parameter or Buffer
            attr = getattr(self.gm, node.target)
            is_buffer = any(node.target == name for name, _ in self.gm.named_buffers())
            if isinstance(attr, torch.Tensor):
                tensor = Tensor(
                    shape=tuple(attr.shape),
                    dtype=self._get_dtype(attr.dtype),
                    name=node.name,
                    data=attr.detach().cpu().numpy(),
                    is_buffer=is_buffer,
                )
                self.builder.parameters.append(tensor)
                self.node_map[node] = tensor

        elif node.op in ("call_function", "call_method", "call_module"):
            # Operation
            op_type = str(node.target)
            if node.op == "call_method":
                op_type = node.target
            elif node.op == "call_module":
                op_type = self.gm.get_submodule(node.target).__class__.__name__

            inputs = []
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    inputs.append(self.node_map.get(arg))
                else:
                    # Constant argument
                    inputs.append(arg)

            # Mapping common torch ops to ONNX-like names
            from onnx9000.converters.torch.aten_map import ATEN_OP_MAP

            mapped_op = ATEN_OP_MAP.get(op_type, op_type)
            if mapped_op == op_type:
                # Fallback logic
                if "add" in op_type:
                    mapped_op = "Add"
                elif "mul" in op_type:
                    mapped_op = "Mul"
                elif "relu" in op_type:
                    mapped_op = "Relu"

            output_tensor = Tensor(name=node.name)
            node_ir = Node(
                op_type=mapped_op,
                inputs=inputs,
                outputs=[output_tensor],
                attributes=node.kwargs,
                name=f"{mapped_op}_{node.name}",
            )

            # Capture metadata hierarchy
            if "nn_module_stack" in node.meta:
                node_ir.attributes["nn_module_stack"] = str(node.meta["nn_module_stack"])

            self.builder.add_node(node_ir)
            self.node_map[node] = output_tensor

        elif node.op == "output":
            # Output
            out_args = node.args[0]
            if isinstance(out_args, torch.fx.Node):
                self.builder.outputs.append(self.node_map.get(out_args))
            elif isinstance(out_args, (list, tuple)):
                for arg in out_args:
                    if isinstance(arg, torch.fx.Node):
                        self.builder.outputs.append(self.node_map.get(arg))
