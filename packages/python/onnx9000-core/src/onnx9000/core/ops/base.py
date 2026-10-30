"""Base helper functions for recording ONNX operators."""

from collections.abc import Sequence

from onnx9000.core.ir import AttributeValue, Tensor, ValueInfo


def record_op(
    op_type: str,
    inputs: Sequence[Tensor | ValueInfo | int | float | str | bool | None],
    attributes: dict[str, AttributeValue] | None = None,
) -> Tensor:
    """Record an ONNX operation and return a symbolic output tensor.

    Args:
        op_type: The ONNX operator type name.
        inputs: Ordered sequence of input tensors or scalar values.
        attributes: Optional dictionary of operator attributes.

    Returns:
        A symbolic Tensor representing the operator output.

    """
    _ = attributes or {}
    tensors = [inp for inp in inputs if isinstance(inp, (Tensor, ValueInfo))]
    dtype = tensors[0].dtype if tensors else None
    return Tensor(name=f"{op_type}_out", shape=(), dtype=dtype)
