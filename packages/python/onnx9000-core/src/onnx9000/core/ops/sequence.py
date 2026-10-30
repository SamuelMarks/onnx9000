"""Sequence construction and manipulation ONNX operators."""

from onnx9000.core.ir import Tensor
from onnx9000.core.ops.base import record_op
from onnx9000.core.registry import register_op


@register_op("ai.onnx", "SequenceConstruct")
def sequence_construct(inputs: list[Tensor]) -> Tensor:
    """Compute SequenceConstruct."""
    return record_op("SequenceConstruct", inputs)


@register_op("ai.onnx", "SequenceEmpty")
def sequence_empty(dtype: int) -> Tensor:
    """Compute SequenceEmpty."""
    return record_op("SequenceEmpty", [], {"dtype": dtype})


@register_op("ai.onnx", "SequenceInsert")
def sequence_insert(seq: Tensor, tensor: Tensor, position: Tensor = None) -> Tensor:
    """Compute SequenceInsert."""
    inputs = [seq, tensor]
    if position is not None:
        inputs.append(position)
    return record_op("SequenceInsert", inputs)


@register_op("ai.onnx", "SequenceMap")
def sequence_map(seq: Tensor) -> Tensor:
    """Compute SequenceMap."""
    return record_op("SequenceMap", [seq])


@register_op("ai.onnx", "ConcatFromSequence")
def concat_from_sequence(seq: Tensor, axis: int, new_axis: int = 0) -> Tensor:
    """Compute ConcatFromSequence."""
    return record_op("ConcatFromSequence", [seq], {"axis": axis, "new_axis": new_axis})


@register_op("ai.onnx", "ReverseSequence")
def reverse_sequence(
    input: Tensor, sequence_lens: Tensor, batch_axis: int = 1, time_axis: int = 0
) -> Tensor:
    """Compute ReverseSequence."""
    return record_op(
        "ReverseSequence",
        [input, sequence_lens],
        {"batch_axis": batch_axis, "time_axis": time_axis},
    )


@register_op("ai.onnx", "SequenceAt")
def sequence_at(input_sequence: Tensor, position: Tensor) -> Tensor:
    """Compute SequenceAt."""
    return record_op("SequenceAt", [input_sequence, position])


@register_op("ai.onnx", "SplitToSequence")
def split_to_sequence(
    input: Tensor, split: Tensor = None, axis: int = 0, keepdims: int = 1
) -> Tensor:
    """Compute SplitToSequence."""
    inputs = [input]
    if split is not None:
        inputs.append(split)
    return record_op("SplitToSequence", inputs, {"axis": axis, "keepdims": keepdims})


@register_op("ai.onnx", "SequenceErase")
def sequence_erase(input_sequence: Tensor, position: Tensor = None) -> Tensor:
    """Compute SequenceErase."""
    inputs = [input_sequence]
    if position is not None:
        inputs.append(position)
    return record_op("SequenceErase", inputs)


@register_op("ai.onnx", "SequenceLength")
def sequence_length(input_sequence: Tensor) -> Tensor:
    """Compute SequenceLength."""
    return record_op("SequenceLength", [input_sequence])
