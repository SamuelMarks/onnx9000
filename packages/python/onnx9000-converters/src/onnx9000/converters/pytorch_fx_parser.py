"""Pytorch fx parser module."""

import json
from typing import Any

from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.registry import global_registry

# Define standard mappings for PyTorch ATen ops to ONNX ops using the registry
# In a real setup, these would be properly decorated functions, but we can do it here inline for the parser


def _map_aten_add_Tensor(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map aten add tensor."""
    return Node(op_type="Add", inputs=inputs, outputs=outputs, name=f"add_{outputs[0]}")


def _map_aten_mul_Tensor(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map aten mul tensor."""
    return Node(op_type="Mul", inputs=inputs, outputs=outputs, name=f"mul_{outputs[0]}")


def _map_aten_convolution_default(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """map aten convolution default."""
    return Node(op_type="Conv", inputs=inputs, outputs=outputs, name=f"conv_{outputs[0]}")


def _map_aten_native_batch_norm_legit_no_training_default(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """map aten native batch norm legit no training default."""
    return Node(
        op_type="BatchNorm", inputs=inputs, outputs=outputs, name=f"batch_norm_{outputs[0]}"
    )


def _map_aten_native_layer_norm_default(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """map aten native layer norm default."""
    return Node(
        op_type="LayerNorm", inputs=inputs, outputs=outputs, name=f"layer_norm_{outputs[0]}"
    )


def _map_aten_bmm_default(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map aten bmm default."""
    return Node(op_type="MatMul", inputs=inputs, outputs=outputs, name=f"bmm_{outputs[0]}")


def _map_aten_mm_default(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map aten mm default."""
    return Node(op_type="MatMul", inputs=inputs, outputs=outputs, name=f"mm_{outputs[0]}")


def _map_aten_max_pool2d_with_indices_default(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    # Discard indices if unused by returning only first output normally mapped
    """map aten max pool2d with indices default."""
    return Node(
        op_type="MaxPool2D", inputs=inputs, outputs=outputs, name=f"max_pool2d_{outputs[0]}"
    )


def _map_aten_gelu_default(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map aten gelu default."""
    return Node(op_type="Gelu", inputs=inputs, outputs=outputs, name=f"gelu_{outputs[0]}")


def _map_aten_arange_start_step(
    inputs: list[str], outputs: list[str], params: dict[str, Any]
) -> Node:
    """map aten arange start step."""
    return Node(op_type="Range", inputs=inputs, outputs=outputs, name=f"arange_{outputs[0]}")


def _map_aten_where_self(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map aten where self."""
    return Node(op_type="Where", inputs=inputs, outputs=outputs, name=f"where_{outputs[0]}")


def _map_aten_copy_(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map aten copy ."""
    return Node(op_type="Identity", inputs=inputs, outputs=outputs, name=f"copy_{outputs[0]}")


def _map_aten_add_(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
    """map aten add ."""
    return Node(op_type="Add", inputs=inputs, outputs=outputs, name=f"add_{outputs[0]}")


# Register them so the parser can use the global_registry as intended by the architecture
from onnx9000.core.registry import register_op

register_op("aten.add.Tensor", "pytorch")(_map_aten_add_Tensor)
register_op("aten.mul.Tensor", "pytorch")(_map_aten_mul_Tensor)
register_op("aten.convolution.default", "pytorch")(_map_aten_convolution_default)
register_op("aten._native_batch_norm_legit_no_training.default", "pytorch")(
    _map_aten_native_batch_norm_legit_no_training_default
)
register_op("aten.native_layer_norm.default", "pytorch")(_map_aten_native_layer_norm_default)
register_op("aten.bmm.default", "pytorch")(_map_aten_bmm_default)
register_op("aten.mm.default", "pytorch")(_map_aten_mm_default)
register_op("aten.max_pool2d_with_indices.default", "pytorch")(
    _map_aten_max_pool2d_with_indices_default
)
register_op("aten.gelu.default", "pytorch")(_map_aten_gelu_default)
register_op("aten.arange.start_step", "pytorch")(_map_aten_arange_start_step)
register_op("aten.where.self", "pytorch")(_map_aten_where_self)
register_op("aten.copy_", "pytorch")(_map_aten_copy_)
register_op("aten.add_", "pytorch")(_map_aten_add_)


class PyTorchFXParser:
    """Parses PyTorch ExportedProgram JSON."""

    def parse_json(self, export_json_str: str) -> Graph:
        """Docstring for D102."""
        data = json.loads(export_json_str)
        graph = Graph(name="PyTorch_FX_Exported")

        for node_data in data.get("nodes", []):
            op = node_data.get("op")
            target = node_data.get("target")
            args = node_data.get("args", [])
            out = node_data.get("out")

            if op == "placeholder":
                # Assuming out is the name of the tensor
                # We need to construct a Tensor properly. For now we use dummy types/shapes
                t = Tensor(name=out, shape=(1,), dtype=1)
                graph.inputs.append(t)
                graph.tensors[out] = t

            elif op == "call_function":
                try:
                    op_func = global_registry.get_op(target, "pytorch")
                    ir_node = op_func(inputs=args, outputs=[out] if out else [], params={})
                except Exception:
                    ir_node = Node(
                        op_type=target,
                        inputs=args,
                        outputs=[out] if out else [],
                        attributes={},
                        name=out or target,
                    )
                graph.nodes.append(ir_node)

            elif op == "output":
                if args and len(args) > 0:
                    for arg in args[0]:  # usually a tuple of outputs
                        if isinstance(arg, str):
                            # Ensure it's in tensors, dummy tensor
                            if arg not in graph.tensors:
                                graph.tensors[arg] = Tensor(name=arg, shape=(1,), dtype=1)
                            graph.outputs.append(graph.tensors[arg])

        return graph


def load_pytorch_fx(json_str: str) -> Graph:
    """Docstring for D103."""
    parser = PyTorchFXParser()
    return parser.parse_json(json_str)
