"""PyTorch ATen and TorchScript operator registrations."""

from typing import Any

from onnx9000.converters.torch.aten_map import ATEN_OP_MAP
from onnx9000.core.ir import Node
from onnx9000.core.registry import register_op


def _create_mapper(op_type: str):
    def mapper(inputs: list[str], outputs: list[str], params: dict[str, Any]) -> Node:
        return Node(
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            attributes=params,
            name=f"{op_type}_{outputs[0]}" if outputs else op_type,
        )

    return mapper


# Register both the aten string format and the direct ONNX format for fallback
for aten_op, onnx_op in ATEN_OP_MAP.items():
    register_op(aten_op, "torch")(_create_mapper(onnx_op))
    register_op(onnx_op, "torch")(_create_mapper(onnx_op))

# Handle some known aliases
register_op("add", "torch")(_create_mapper("Add"))
register_op("mul", "torch")(_create_mapper("Mul"))
register_op("relu", "torch")(_create_mapper("Relu"))
