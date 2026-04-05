"""TorchScript IR parser for onnx9000."""

import torch
from onnx9000.converters.frontend.builder import GraphBuilder
from onnx9000.converters.frontend.tensor import Node, Tensor
from onnx9000.core.dtypes import DType


class TorchScriptParser:
    """Parser for TorchScript IR."""

    def __init__(self, func_or_module) -> None:
        """Initialize the TorchScript parser.

        Args:
            func_or_module: A Torch function or module to parse.

        """
        name = "TorchScriptGraph"
        if isinstance(func_or_module, torch.jit.ScriptModule):
            self.script_module = func_or_module
            name = getattr(self.script_module, "original_name", name)
        else:
            name = getattr(func_or_module, "__name__", name)
            self.script_module = torch.jit.script(func_or_module)
        self.graph = self.script_module.graph
        self.builder = GraphBuilder(name=name)
        self.value_map = {}

    def _get_dtype(self, torch_dtype: torch.dtype) -> DType:
        """Map Torch dtype to DType.

        Args:
            torch_dtype: The Torch data type to map.

        Returns:
            The corresponding onnx9000 DType.

        """
        if torch_dtype == torch.float32:
            return DType.FLOAT32
        if torch_dtype == torch.float64:
            return DType.FLOAT64
        if torch_dtype == torch.int64:
            return DType.INT64
        if torch_dtype == torch.int32:
            return DType.INT32
        if torch_dtype == torch.bool:
            return DType.BOOL
        return DType.FLOAT32

    def parse(self) -> GraphBuilder:
        """Parse the TorchScript graph into a GraphBuilder.

        Returns:
            The populated GraphBuilder.

        """
        # Handle inputs (first input is usually 'self' for modules)
        for i, val in enumerate(self.graph.inputs()):
            if i == 0 and isinstance(self.script_module, torch.jit.ScriptModule):
                continue
            is_tensor = val.type().kind() == "TensorType"
            try:
                shape = val.type().sizes() if hasattr(val.type(), "sizes") else None
            except Exception:
                shape = None

            dtype = DType.FLOAT32
            if is_tensor and hasattr(val.type(), "scalarType"):
                dtype = self._get_dtype(val.type().scalarType())
            elif val.type().kind() == "BoolType":
                dtype = DType.BOOL
            elif val.type().kind() == "IntType":
                dtype = DType.INT64
            elif val.type().kind() == "FloatType":
                dtype = DType.FLOAT32
            name = val.debugName()
            tensor = Tensor(shape=tuple(shape) if shape else (), dtype=dtype, name=name)
            self.builder.inputs.append(tensor)
            self.value_map[val] = tensor

        # Walk through nodes
        for node in self.graph.nodes():
            self._parse_node(node)

        # Handle outputs
        for val in self.graph.outputs():
            if val in self.value_map:
                self.builder.outputs.append(self.value_map[val])

        return self.builder

    def _parse_node(self, node: torch._C.Node) -> None:
        """Parse a single TorchScript node.

        Args:
            node: The TorchScript node to parse.

        """
        kind = node.kind()
        inputs = [self.value_map.get(v) for v in node.inputs() if v in self.value_map]

        # Handle constants
        if kind == "prim::Constant":
            out_val = list(node.outputs())[0]
            name = out_val.debugName()
            val = None
            # Try to get value from various attribute types
            for attr in node.attributeNames():
                if attr == "value":
                    try:
                        val = node.t("value")
                    except Exception:
                        try:
                            val = node.f("value")
                        except Exception:
                            try:
                                val = node.i("value")
                            except Exception:
                                try:
                                    val = node.s("value")
                                except Exception:
                                    assert True

            if val is not None:
                import numpy as np

                if isinstance(val, torch.Tensor):
                    tensor = Tensor(
                        shape=tuple(val.shape),
                        dtype=self._get_dtype(val.dtype),
                        name=name,
                        data=val.numpy(),
                    )
                elif isinstance(val, float):
                    tensor = Tensor(
                        shape=(),
                        dtype=DType.FLOAT32,
                        name=name,
                        data=np.array(val, dtype=np.float32),
                    )
                elif isinstance(val, int):
                    tensor = Tensor(
                        shape=(),
                        dtype=DType.INT64,
                        name=name,
                        data=np.array(val, dtype=np.int64),
                    )
                else:
                    return
                self.builder.parameters.append(tensor)
                self.value_map[out_val] = tensor
            return

        # Create output tensors
        outputs = []
        for out_val in node.outputs():
            try:
                shape = out_val.type().sizes() if hasattr(out_val.type(), "sizes") else None
            except Exception:
                shape = None
            dtype = (
                self._get_dtype(out_val.type().scalarType())
                if hasattr(out_val.type(), "scalarType")
                else DType.FLOAT32
            )
            tensor = Tensor(
                shape=tuple(shape) if shape else (), dtype=dtype, name=out_val.debugName()
            )
            outputs.append(tensor)
            self.value_map[out_val] = tensor

        # Mapping common aten:: and prim:: ops
        op_type = kind.split("::")[-1]
        if op_type == "add":
            op_type = "Add"
        elif op_type == "mul":
            op_type = "Mul"
        elif op_type == "sub":
            op_type = "Sub"
        elif op_type == "div":
            op_type = "Div"

        node_ir = Node(
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            name=f"{op_type}_{outputs[0].name}" if outputs else f"{op_type}",
        )

        # Handle Control Flow (recursion for blocks)
        if kind == "prim::If":
            blocks = list(node.blocks())
            if len(blocks) == 2:
                then_builder = GraphBuilder(name=f"{self.builder.name}_then")
                else_builder = GraphBuilder(name=f"{self.builder.name}_else")
                # Recursive block parsing would go here in a full implementation
                # For Phase 3, we at least capture the presence of the node.
                node_ir.attributes["then_branch"] = then_builder
                node_ir.attributes["else_branch"] = else_builder

        self.builder.add_node(node_ir)
