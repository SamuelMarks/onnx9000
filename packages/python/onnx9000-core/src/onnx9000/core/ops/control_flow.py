"""Control flow, conditional, optional, and collective ONNX operators."""

from onnx9000.core.ir import AttributeValue, Tensor
from onnx9000.core.ops.base import record_op
from onnx9000.core.registry import register_op


@register_op("ai.onnx", "Where")
def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Compute Where."""
    return record_op("Where", [condition, x, y])


@register_op("onnx9000.custom", "AllReduce")
def all_reduce(x: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute Collective AllReduce."""
    return record_op("AllReduce", [x], kwargs)


@register_op("onnx9000.custom", "AllGather")
def all_gather(x: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute Collective AllGather."""
    return record_op("AllGather", [x], kwargs)


@register_op("onnx9000.custom", "ReduceScatter")
def reduce_scatter(x: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute Collective ReduceScatter."""
    return record_op("ReduceScatter", [x], kwargs)


@register_op("onnx9000.custom", "Broadcast")
def broadcast(x: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute Collective Broadcast."""
    return record_op("Broadcast", [x], kwargs)


@register_op("ai.onnx", "If")
def if_op(x: Tensor) -> Tensor:
    """Compute If."""
    return record_op("If", [x])


@register_op("ai.onnx", "Loop")
def loop(x: Tensor) -> Tensor:
    """Compute Loop."""
    return record_op("Loop", [x])


@register_op("ai.onnx", "Optional")
def optional(x: Tensor) -> Tensor:
    """Compute Optional."""
    return record_op("Optional", [x])


@register_op("ai.onnx", "OptionalGetElement")
def optional_get_element(x: Tensor) -> Tensor:
    """Compute OptionalGetElement."""
    return record_op("OptionalGetElement", [x])


@register_op("ai.onnx", "OptionalHasElement")
def optional_has_element(x: Tensor) -> Tensor:
    """Compute OptionalHasElement."""
    return record_op("OptionalHasElement", [x])


@register_op("ai.onnx", "Scan")
def scan(x: Tensor) -> Tensor:
    """Compute Scan."""
    return record_op("Scan", [x])
