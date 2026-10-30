"""PyTorch-compatible indexing, slicing, view, and shape manipulation operations."""

from onnx9000.core.ir import AttributeValue, Tensor


def chunk(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply chunk to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for chunk.
    """
    from onnx9000.core.ops.torch_auto import chunk as core_chunk

    return core_chunk(input_tensor, *args, **kwargs)


def split(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply split to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for split.
    """
    from onnx9000.core.ops.torch_auto import split as core_split

    return core_split(input_tensor, *args, **kwargs)


def stack(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply stack to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for stack.
    """
    from onnx9000.core.ops.torch_auto import stack as core_stack

    return core_stack(input_tensor, *args, **kwargs)


def unravel_index(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply unravel_index to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for unravel_index.
    """
    from onnx9000.core.ops.torch_auto import unravel_index as core_unravel_index

    return core_unravel_index(input_tensor, *args, **kwargs)


def alias_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply alias_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for alias_copy.
    """
    from onnx9000.core.ops.torch_auto import alias_copy as core_alias_copy

    return core_alias_copy(input_tensor, *args, **kwargs)


def align_tensors(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply align_tensors to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for align_tensors.
    """
    from onnx9000.core.ops.torch_auto import align_tensors as core_align_tensors

    return core_align_tensors(input_tensor, *args, **kwargs)


def argsort(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply argsort to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for argsort.
    """
    from onnx9000.core.ops.torch_auto import argsort as core_argsort

    return core_argsort(input_tensor, *args, **kwargs)


def argwhere(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply argwhere to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for argwhere.
    """
    from onnx9000.core.ops.torch_auto import argwhere as core_argwhere

    return core_argwhere(input_tensor, *args, **kwargs)


def as_strided(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply as_strided to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for as_strided.
    """
    from onnx9000.core.ops.torch_auto import as_strided as core_as_strided

    return core_as_strided(input_tensor, *args, **kwargs)


def as_strided_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply as_strided_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for as_strided_.
    """
    from onnx9000.core.ops.torch_auto import as_strided_ as core_as_strided_

    return core_as_strided_(input_tensor, *args, **kwargs)


def as_strided_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply as_strided_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for as_strided_copy.
    """
    from onnx9000.core.ops.torch_auto import as_strided_copy as core_as_strided_copy

    return core_as_strided_copy(input_tensor, *args, **kwargs)


def as_strided_scatter(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply as_strided_scatter to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for as_strided_scatter.
    """
    from onnx9000.core.ops.torch_auto import as_strided_scatter as core_as_strided_scatter

    return core_as_strided_scatter(input_tensor, *args, **kwargs)


def atleast_1d(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply atleast_1d to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for atleast_1d.
    """
    from onnx9000.core.ops.torch_auto import atleast_1d as core_atleast_1d

    return core_atleast_1d(input_tensor, *args, **kwargs)


def atleast_2d(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply atleast_2d to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for atleast_2d.
    """
    from onnx9000.core.ops.torch_auto import atleast_2d as core_atleast_2d

    return core_atleast_2d(input_tensor, *args, **kwargs)


def atleast_3d(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply atleast_3d to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for atleast_3d.
    """
    from onnx9000.core.ops.torch_auto import atleast_3d as core_atleast_3d

    return core_atleast_3d(input_tensor, *args, **kwargs)


def block_diag(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply block_diag to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for block_diag.
    """
    from onnx9000.core.ops.torch_auto import block_diag as core_block_diag

    return core_block_diag(input_tensor, *args, **kwargs)


def broadcast_tensors(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply broadcast_tensors to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for broadcast_tensors.
    """
    from onnx9000.core.ops.torch_auto import broadcast_tensors as core_broadcast_tensors

    return core_broadcast_tensors(input_tensor, *args, **kwargs)


def broadcast_to(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply broadcast_to to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for broadcast_to.
    """
    from onnx9000.core.ops.torch_auto import broadcast_to as core_broadcast_to

    return core_broadcast_to(input_tensor, *args, **kwargs)


def bucketize(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply bucketize to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for bucketize.
    """
    from onnx9000.core.ops.torch_auto import bucketize as core_bucketize

    return core_bucketize(input_tensor, *args, **kwargs)


def cartesian_prod(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cartesian_prod to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cartesian_prod.
    """
    from onnx9000.core.ops.torch_auto import cartesian_prod as core_cartesian_prod

    return core_cartesian_prod(input_tensor, *args, **kwargs)


def cat(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cat to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cat.
    """
    from onnx9000.core.ops.torch_auto import cat as core_cat

    return core_cat(input_tensor, *args, **kwargs)


def ccol_indices_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply ccol_indices_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for ccol_indices_copy.
    """
    from onnx9000.core.ops.torch_auto import ccol_indices_copy as core_ccol_indices_copy

    return core_ccol_indices_copy(input_tensor, *args, **kwargs)


def clone(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply clone to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for clone.
    """
    from onnx9000.core.ops.torch_auto import clone as core_clone

    return core_clone(input_tensor, *args, **kwargs)


def col_indices_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply col_indices_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for col_indices_copy.
    """
    from onnx9000.core.ops.torch_auto import col_indices_copy as core_col_indices_copy

    return core_col_indices_copy(input_tensor, *args, **kwargs)


def column_stack(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply column_stack to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for column_stack.
    """
    from onnx9000.core.ops.torch_auto import column_stack as core_column_stack

    return core_column_stack(input_tensor, *args, **kwargs)


def combinations(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply combinations to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for combinations.
    """
    from onnx9000.core.ops.torch_auto import combinations as core_combinations

    return core_combinations(input_tensor, *args, **kwargs)


def concatenate(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply concatenate to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for concatenate.
    """
    from onnx9000.core.ops.torch_auto import concatenate as core_concatenate

    return core_concatenate(input_tensor, *args, **kwargs)


def crow_indices_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply crow_indices_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for crow_indices_copy.
    """
    from onnx9000.core.ops.torch_auto import crow_indices_copy as core_crow_indices_copy

    return core_crow_indices_copy(input_tensor, *args, **kwargs)


def detach(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply detach to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for detach.
    """
    from onnx9000.core.ops.torch_auto import detach as core_detach

    return core_detach(input_tensor, *args, **kwargs)


def detach_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply detach_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for detach_.
    """
    from onnx9000.core.ops.torch_auto import detach_ as core_detach_

    return core_detach_(input_tensor, *args, **kwargs)


def detach_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply detach_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for detach_copy.
    """
    from onnx9000.core.ops.torch_auto import detach_copy as core_detach_copy

    return core_detach_copy(input_tensor, *args, **kwargs)


def dsplit(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply dsplit to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for dsplit.
    """
    from onnx9000.core.ops.torch_auto import dsplit as core_dsplit

    return core_dsplit(input_tensor, *args, **kwargs)


def dstack(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply dstack to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for dstack.
    """
    from onnx9000.core.ops.torch_auto import dstack as core_dstack

    return core_dstack(input_tensor, *args, **kwargs)


def expand_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply expand_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for expand_copy.
    """
    from onnx9000.core.ops.torch_auto import expand_copy as core_expand_copy

    return core_expand_copy(input_tensor, *args, **kwargs)


def flatten(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply flatten to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for flatten.
    """
    from onnx9000.core.ops.torch_auto import flatten as core_flatten

    return core_flatten(input_tensor, *args, **kwargs)


def flip(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply flip to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for flip.
    """
    from onnx9000.core.ops.torch_auto import flip as core_flip

    return core_flip(input_tensor, *args, **kwargs)


def fliplr(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply fliplr to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for fliplr.
    """
    from onnx9000.core.ops.torch_auto import fliplr as core_fliplr

    return core_fliplr(input_tensor, *args, **kwargs)


def flipud(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply flipud to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for flipud.
    """
    from onnx9000.core.ops.torch_auto import flipud as core_flipud

    return core_flipud(input_tensor, *args, **kwargs)


def hsplit(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply hsplit to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for hsplit.
    """
    from onnx9000.core.ops.torch_auto import hsplit as core_hsplit

    return core_hsplit(input_tensor, *args, **kwargs)


def hstack(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply hstack to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for hstack.
    """
    from onnx9000.core.ops.torch_auto import hstack as core_hstack

    return core_hstack(input_tensor, *args, **kwargs)


def index_add(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply index_add to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for index_add.
    """
    from onnx9000.core.ops.torch_auto import index_add as core_index_add

    return core_index_add(input_tensor, *args, **kwargs)


def index_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply index_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for index_copy.
    """
    from onnx9000.core.ops.torch_auto import index_copy as core_index_copy

    return core_index_copy(input_tensor, *args, **kwargs)


def index_fill(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply index_fill to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for index_fill.
    """
    from onnx9000.core.ops.torch_auto import index_fill as core_index_fill

    return core_index_fill(input_tensor, *args, **kwargs)


def index_put(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply index_put to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for index_put.
    """
    from onnx9000.core.ops.torch_auto import index_put as core_index_put

    return core_index_put(input_tensor, *args, **kwargs)


def index_put_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply index_put_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for index_put_.
    """
    from onnx9000.core.ops.torch_auto import index_put_ as core_index_put_

    return core_index_put_(input_tensor, *args, **kwargs)


def index_reduce(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply index_reduce to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for index_reduce.
    """
    from onnx9000.core.ops.torch_auto import index_reduce as core_index_reduce

    return core_index_reduce(input_tensor, *args, **kwargs)


def index_select(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply index_select to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for index_select.
    """
    from onnx9000.core.ops.torch_auto import index_select as core_index_select

    return core_index_select(input_tensor, *args, **kwargs)


def indices_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply indices_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for indices_copy.
    """
    from onnx9000.core.ops.torch_auto import indices_copy as core_indices_copy

    return core_indices_copy(input_tensor, *args, **kwargs)


def isin(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply isin to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for isin.
    """
    from onnx9000.core.ops.torch_auto import isin as core_isin

    return core_isin(input_tensor, *args, **kwargs)


def masked_fill(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply masked_fill to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for masked_fill.
    """
    from onnx9000.core.ops.torch_auto import masked_fill as core_masked_fill

    return core_masked_fill(input_tensor, *args, **kwargs)


def masked_scatter(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply masked_scatter to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for masked_scatter.
    """
    from onnx9000.core.ops.torch_auto import masked_scatter as core_masked_scatter

    return core_masked_scatter(input_tensor, *args, **kwargs)


def masked_select(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply masked_select to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for masked_select.
    """
    from onnx9000.core.ops.torch_auto import masked_select as core_masked_select

    return core_masked_select(input_tensor, *args, **kwargs)


def meshgrid(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply meshgrid to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for meshgrid.
    """
    from onnx9000.core.ops.torch_auto import meshgrid as core_meshgrid

    return core_meshgrid(input_tensor, *args, **kwargs)


def moveaxis(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply moveaxis to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for moveaxis.
    """
    from onnx9000.core.ops.torch_auto import moveaxis as core_moveaxis

    return core_moveaxis(input_tensor, *args, **kwargs)


def movedim(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply movedim to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for movedim.
    """
    from onnx9000.core.ops.torch_auto import movedim as core_movedim

    return core_movedim(input_tensor, *args, **kwargs)


def msort(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply msort to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for msort.
    """
    from onnx9000.core.ops.torch_auto import msort as core_msort

    return core_msort(input_tensor, *args, **kwargs)


def narrow(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply narrow to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for narrow.
    """
    from onnx9000.core.ops.torch_auto import narrow as core_narrow

    return core_narrow(input_tensor, *args, **kwargs)


def narrow_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply narrow_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for narrow_copy.
    """
    from onnx9000.core.ops.torch_auto import narrow_copy as core_narrow_copy

    return core_narrow_copy(input_tensor, *args, **kwargs)


def nonzero(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply nonzero to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for nonzero.
    """
    from onnx9000.core.ops.torch_auto import nonzero as core_nonzero

    return core_nonzero(input_tensor, *args, **kwargs)


def nonzero_static(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply nonzero_static to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for nonzero_static.
    """
    from onnx9000.core.ops.torch_auto import nonzero_static as core_nonzero_static

    return core_nonzero_static(input_tensor, *args, **kwargs)


def permute(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply permute to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for permute.
    """
    from onnx9000.core.ops.torch_auto import permute as core_permute

    return core_permute(input_tensor, *args, **kwargs)


def permute_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply permute_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for permute_copy.
    """
    from onnx9000.core.ops.torch_auto import permute_copy as core_permute_copy

    return core_permute_copy(input_tensor, *args, **kwargs)


def put(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply put to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for put.
    """
    from onnx9000.core.ops.torch_auto import put as core_put

    return core_put(input_tensor, *args, **kwargs)


def ravel(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply ravel to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for ravel.
    """
    from onnx9000.core.ops.torch_auto import ravel as core_ravel

    return core_ravel(input_tensor, *args, **kwargs)


def repeat_interleave(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply repeat_interleave to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for repeat_interleave.
    """
    from onnx9000.core.ops.torch_auto import repeat_interleave as core_repeat_interleave

    return core_repeat_interleave(input_tensor, *args, **kwargs)


def resize_as_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply resize_as_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for resize_as_.
    """
    from onnx9000.core.ops.torch_auto import resize_as_ as core_resize_as_

    return core_resize_as_(input_tensor, *args, **kwargs)


def resize_as_sparse_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply resize_as_sparse_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for resize_as_sparse_.
    """
    from onnx9000.core.ops.torch_auto import resize_as_sparse_ as core_resize_as_sparse_

    return core_resize_as_sparse_(input_tensor, *args, **kwargs)


def roll(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply roll to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for roll.
    """
    from onnx9000.core.ops.torch_auto import roll as core_roll

    return core_roll(input_tensor, *args, **kwargs)


def rot90(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply rot90 to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for rot90.
    """
    from onnx9000.core.ops.torch_auto import rot90 as core_rot90

    return core_rot90(input_tensor, *args, **kwargs)


def row_indices_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply row_indices_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for row_indices_copy.
    """
    from onnx9000.core.ops.torch_auto import row_indices_copy as core_row_indices_copy

    return core_row_indices_copy(input_tensor, *args, **kwargs)


def row_stack(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply row_stack to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for row_stack.
    """
    from onnx9000.core.ops.torch_auto import row_stack as core_row_stack

    return core_row_stack(input_tensor, *args, **kwargs)


def scatter_add(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply scatter_add to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for scatter_add.
    """
    from onnx9000.core.ops.torch_auto import scatter_add as core_scatter_add

    return core_scatter_add(input_tensor, *args, **kwargs)


def scatter_reduce(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply scatter_reduce to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for scatter_reduce.
    """
    from onnx9000.core.ops.torch_auto import scatter_reduce as core_scatter_reduce

    return core_scatter_reduce(input_tensor, *args, **kwargs)


def searchsorted(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply searchsorted to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for searchsorted.
    """
    from onnx9000.core.ops.torch_auto import searchsorted as core_searchsorted

    return core_searchsorted(input_tensor, *args, **kwargs)


def select(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply select to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for select.
    """
    from onnx9000.core.ops.torch_auto import select as core_select

    return core_select(input_tensor, *args, **kwargs)


def select_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply select_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for select_copy.
    """
    from onnx9000.core.ops.torch_auto import select_copy as core_select_copy

    return core_select_copy(input_tensor, *args, **kwargs)


def select_scatter(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply select_scatter to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for select_scatter.
    """
    from onnx9000.core.ops.torch_auto import select_scatter as core_select_scatter

    return core_select_scatter(input_tensor, *args, **kwargs)


def slice_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply slice_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for slice_copy.
    """
    from onnx9000.core.ops.torch_auto import slice_copy as core_slice_copy

    return core_slice_copy(input_tensor, *args, **kwargs)


def slice_inverse(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply slice_inverse to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for slice_inverse.
    """
    from onnx9000.core.ops.torch_auto import slice_inverse as core_slice_inverse

    return core_slice_inverse(input_tensor, *args, **kwargs)


def slice_scatter(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply slice_scatter to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for slice_scatter.
    """
    from onnx9000.core.ops.torch_auto import slice_scatter as core_slice_scatter

    return core_slice_scatter(input_tensor, *args, **kwargs)


def sort(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply sort to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for sort.
    """
    from onnx9000.core.ops.torch_auto import sort as core_sort

    return core_sort(input_tensor, *args, **kwargs)


def split_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply split_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for split_copy.
    """
    from onnx9000.core.ops.torch_auto import split_copy as core_split_copy

    return core_split_copy(input_tensor, *args, **kwargs)


def split_with_sizes(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply split_with_sizes to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for split_with_sizes.
    """
    from onnx9000.core.ops.torch_auto import split_with_sizes as core_split_with_sizes

    return core_split_with_sizes(input_tensor, *args, **kwargs)


def split_with_sizes_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply split_with_sizes_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for split_with_sizes_copy.
    """
    from onnx9000.core.ops.torch_auto import split_with_sizes_copy as core_split_with_sizes_copy

    return core_split_with_sizes_copy(input_tensor, *args, **kwargs)


def squeeze(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply squeeze to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for squeeze.
    """
    from onnx9000.core.ops.torch_auto import squeeze as core_squeeze

    return core_squeeze(input_tensor, *args, **kwargs)


def squeeze_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply squeeze_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for squeeze_copy.
    """
    from onnx9000.core.ops.torch_auto import squeeze_copy as core_squeeze_copy

    return core_squeeze_copy(input_tensor, *args, **kwargs)


def swapaxes(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply swapaxes to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for swapaxes.
    """
    from onnx9000.core.ops.torch_auto import swapaxes as core_swapaxes

    return core_swapaxes(input_tensor, *args, **kwargs)


def swapdims(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply swapdims to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for swapdims.
    """
    from onnx9000.core.ops.torch_auto import swapdims as core_swapdims

    return core_swapdims(input_tensor, *args, **kwargs)


def t(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply t to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for t.
    """
    from onnx9000.core.ops.torch_auto import t as core_t

    return core_t(input_tensor, *args, **kwargs)


def t_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply t_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for t_copy.
    """
    from onnx9000.core.ops.torch_auto import t_copy as core_t_copy

    return core_t_copy(input_tensor, *args, **kwargs)


def take(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply take to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for take.
    """
    from onnx9000.core.ops.torch_auto import take as core_take

    return core_take(input_tensor, *args, **kwargs)


def take_along_dim(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply take_along_dim to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for take_along_dim.
    """
    from onnx9000.core.ops.torch_auto import take_along_dim as core_take_along_dim

    return core_take_along_dim(input_tensor, *args, **kwargs)


def tensor_split(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply tensor_split to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for tensor_split.
    """
    from onnx9000.core.ops.torch_auto import tensor_split as core_tensor_split

    return core_tensor_split(input_tensor, *args, **kwargs)


def transpose_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply transpose_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for transpose_copy.
    """
    from onnx9000.core.ops.torch_auto import transpose_copy as core_transpose_copy

    return core_transpose_copy(input_tensor, *args, **kwargs)


def unbind(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply unbind to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for unbind.
    """
    from onnx9000.core.ops.torch_auto import unbind as core_unbind

    return core_unbind(input_tensor, *args, **kwargs)


def unbind_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply unbind_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for unbind_copy.
    """
    from onnx9000.core.ops.torch_auto import unbind_copy as core_unbind_copy

    return core_unbind_copy(input_tensor, *args, **kwargs)


def unflatten(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply unflatten to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for unflatten.
    """
    from onnx9000.core.ops.torch_auto import unflatten as core_unflatten

    return core_unflatten(input_tensor, *args, **kwargs)


def unfold_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply unfold_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for unfold_copy.
    """
    from onnx9000.core.ops.torch_auto import unfold_copy as core_unfold_copy

    return core_unfold_copy(input_tensor, *args, **kwargs)


def unsafe_chunk(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply unsafe_chunk to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for unsafe_chunk.
    """
    from onnx9000.core.ops.torch_auto import unsafe_chunk as core_unsafe_chunk

    return core_unsafe_chunk(input_tensor, *args, **kwargs)


def unsafe_split(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply unsafe_split to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for unsafe_split.
    """
    from onnx9000.core.ops.torch_auto import unsafe_split as core_unsafe_split

    return core_unsafe_split(input_tensor, *args, **kwargs)


def unsafe_split_with_sizes(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply unsafe_split_with_sizes to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for unsafe_split_with_sizes.
    """
    from onnx9000.core.ops.torch_auto import unsafe_split_with_sizes as core_unsafe_split_with_sizes

    return core_unsafe_split_with_sizes(input_tensor, *args, **kwargs)


def unsqueeze(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply unsqueeze to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for unsqueeze.
    """
    from onnx9000.core.ops.torch_auto import unsqueeze as core_unsqueeze

    return core_unsqueeze(input_tensor, *args, **kwargs)


def unsqueeze_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply unsqueeze_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for unsqueeze_copy.
    """
    from onnx9000.core.ops.torch_auto import unsqueeze_copy as core_unsqueeze_copy

    return core_unsqueeze_copy(input_tensor, *args, **kwargs)


def values_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply values_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for values_copy.
    """
    from onnx9000.core.ops.torch_auto import values_copy as core_values_copy

    return core_values_copy(input_tensor, *args, **kwargs)


def view_as_complex(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply view_as_complex to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for view_as_complex.
    """
    from onnx9000.core.ops.torch_auto import view_as_complex as core_view_as_complex

    return core_view_as_complex(input_tensor, *args, **kwargs)


def view_as_complex_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply view_as_complex_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for view_as_complex_copy.
    """
    from onnx9000.core.ops.torch_auto import view_as_complex_copy as core_view_as_complex_copy

    return core_view_as_complex_copy(input_tensor, *args, **kwargs)


def view_as_real(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply view_as_real to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for view_as_real.
    """
    from onnx9000.core.ops.torch_auto import view_as_real as core_view_as_real

    return core_view_as_real(input_tensor, *args, **kwargs)


def view_as_real_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply view_as_real_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for view_as_real_copy.
    """
    from onnx9000.core.ops.torch_auto import view_as_real_copy as core_view_as_real_copy

    return core_view_as_real_copy(input_tensor, *args, **kwargs)


def view_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply view_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for view_copy.
    """
    from onnx9000.core.ops.torch_auto import view_copy as core_view_copy

    return core_view_copy(input_tensor, *args, **kwargs)


def vsplit(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply vsplit to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for vsplit.
    """
    from onnx9000.core.ops.torch_auto import vsplit as core_vsplit

    return core_vsplit(input_tensor, *args, **kwargs)


def vstack(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply vstack to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for vstack.
    """
    from onnx9000.core.ops.torch_auto import vstack as core_vstack

    return core_vstack(input_tensor, *args, **kwargs)
