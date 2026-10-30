"""PyTorch ATen and TorchScript operator registrations."""

from onnx9000.converters.torch.aten_map import ATEN_OP_MAP
from onnx9000.core.ir import Node
from onnx9000.core.registry import register_op


def _create_mapper(op_type: str) -> int | float | str | bool | bytes | list | dict | tuple | None:
    """Execute create mapper operation.

    Args:
        op_type: Parameter specifying op type (str).

    Returns:
        Result of create mapper.
    """

    def mapper(
        inputs: list[str],
        outputs: list[str],
        params: dict[str, int | float | str | bool | bytes | list | dict | tuple | None],
    ) -> Node:
        """Execute mapper operation.

        Args:
            inputs: Parameter specifying inputs (list[str]).
            outputs: Parameter specifying outputs (list[str]).
            params: Parameter specifying params (dict[str, Union[int, float, str, bool, bytes, list, dict, tuple, None]]).

        Returns:
            Result of mapper (Node).
        """
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
    register_op("torch", aten_op)(_create_mapper(onnx_op))
    register_op("torch", onnx_op)(_create_mapper(onnx_op))

# Handle some known aliases
register_op("torch", "add")(_create_mapper("Add"))
register_op("torch", "mul")(_create_mapper("Mul"))
register_op("torch", "relu")(_create_mapper("Relu"))
