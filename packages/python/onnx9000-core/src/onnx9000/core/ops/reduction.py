"""Reduction and aggregation ONNX operators."""

from onnx9000.core.ir import AttributeValue, Tensor
from onnx9000.core.ops.base import record_op
from onnx9000.core.registry import register_op


@register_op("ai.onnx", "Mean")
def mean(inputs: list[Tensor]) -> Tensor:
    """Compute Mean."""
    return record_op("Mean", inputs)


@register_op("ai.onnx", "ReduceSumSquare")
def reduce_sum_square(data: Tensor, axes: list[int] | None = None, keepdims: int = 1) -> Tensor:
    """Compute ReduceSumSquare."""
    attrs = {"keepdims": keepdims}
    if axes is not None:
        attrs["axes"] = axes
    return record_op("ReduceSumSquare", [data], attrs)


@register_op("ai.onnx", "ReduceL1")
def reduce_l1(data: Tensor, axes: list[int] | None = None, keepdims: int = 1) -> Tensor:
    """Compute ReduceL1."""
    attrs = {"keepdims": keepdims}
    if axes is not None:
        attrs["axes"] = axes
    return record_op("ReduceL1", [data], attrs)


@register_op("ai.onnx", "ReduceL2")
def reduce_l2(data: Tensor, axes: list[int] | None = None, keepdims: int = 1) -> Tensor:
    """Compute ReduceL2."""
    attrs = {"keepdims": keepdims}
    if axes is not None:
        attrs["axes"] = axes
    return record_op("ReduceL2", [data], attrs)


@register_op("ai.onnx", "ReduceLogSum")
def reduce_log_sum(data: Tensor, axes: list[int] | None = None, keepdims: int = 1) -> Tensor:
    """Compute ReduceLogSum."""
    attrs = {"keepdims": keepdims}
    if axes is not None:
        attrs["axes"] = axes
    return record_op("ReduceLogSum", [data], attrs)


@register_op("ai.onnx", "ReduceLogSumExp")
def reduce_log_sum_exp(data: Tensor, axes: list[int] | None = None, keepdims: int = 1) -> Tensor:
    """Compute ReduceLogSumExp."""
    attrs = {"keepdims": keepdims}
    if axes is not None:
        attrs["axes"] = axes
    return record_op("ReduceLogSumExp", [data], attrs)


@register_op("ai.onnx", "TopK")
def topk(X: Tensor, K: Tensor, axis: int = -1, largest: int = 1, sorted: int = 1) -> Tensor:
    """Compute TopK."""
    return record_op("TopK", [X, K], {"axis": axis, "largest": largest, "sorted": sorted})


@register_op("ai.onnx", "Argmax")
def argmax(data: Tensor, axis: int = 0, keepdims: int = 1, select_last_index: int = 0) -> Tensor:
    """Compute ArgMax."""
    return record_op(
        "ArgMax",
        [data],
        {"axis": axis, "keepdims": keepdims, "select_last_index": select_last_index},
    )


@register_op("ai.onnx", "Argmin")
def argmin(data: Tensor, axis: int = 0, keepdims: int = 1, select_last_index: int = 0) -> Tensor:
    """Compute ArgMin."""
    return record_op(
        "ArgMin",
        [data],
        {"axis": axis, "keepdims": keepdims, "select_last_index": select_last_index},
    )


@register_op("ai.onnx", "CumSum")
def cumsum(x: Tensor, axis: Tensor, exclusive: int = 0, reverse: int = 0) -> Tensor:
    """Compute CumSum."""
    return record_op("CumSum", [x, axis], {"exclusive": exclusive, "reverse": reverse})


@register_op("ai.onnx", "ReduceSum")
def reduce_sum(x: Tensor, axes: list[int] | None = None, keepdims: bool = True) -> Tensor:
    """Compute the sum of the input tensor's element along the provided axes."""
    attributes: dict[str, AttributeValue] = {"keepdims": 1 if keepdims else 0}
    if axes is not None:
        attributes["axes"] = axes
    return record_op("ReduceSum", [x], attributes)


@register_op("ai.onnx", "ReduceMean")
def reduce_mean(x: Tensor, axes: list[int] | None = None, keepdims: bool = True) -> Tensor:
    """Compute the mean of the input tensor's element along the provided axes."""
    attributes: dict[str, AttributeValue] = {"keepdims": 1 if keepdims else 0}
    if axes is not None:
        attributes["axes"] = axes
    return record_op("ReduceMean", [x], attributes)


@register_op("ai.onnx", "ReduceMax")
def reduce_max(x: Tensor, axes: list[int] | None = None, keepdims: bool = True) -> Tensor:
    """Compute ReduceMax."""
    attributes: dict[str, AttributeValue] = {"keepdims": 1 if keepdims else 0}
    if axes is not None:
        attributes["axes"] = axes
    return record_op("ReduceMax", [x], attributes)


@register_op("ai.onnx", "ReduceMin")
def reduce_min(x: Tensor, axes: list[int] | None = None, keepdims: bool = True) -> Tensor:
    """Compute ReduceMin."""
    attributes: dict[str, AttributeValue] = {"keepdims": 1 if keepdims else 0}
    if axes is not None:
        attributes["axes"] = axes
    return record_op("ReduceMin", [x], attributes)


@register_op("ai.onnx", "ReduceProd")
def reduce_prod(x: Tensor, axes: list[int] | None = None, keepdims: bool = True) -> Tensor:
    """Compute ReduceProd."""
    attributes: dict[str, AttributeValue] = {"keepdims": 1 if keepdims else 0}
    if axes is not None:
        attributes["axes"] = axes
    return record_op("ReduceProd", [x], attributes)


@register_op("ai.onnx", "Sum")
def sum(data: list[Tensor]) -> Tensor:
    """Compute Sum."""
    return record_op("Sum", data)


@register_op("ai.onnx", "CumProd")
def cum_prod(x: Tensor) -> Tensor:
    """Compute CumProd."""
    return record_op("CumProd", [x])
