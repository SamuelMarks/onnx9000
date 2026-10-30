"""PyTorch-compatible tensor creation and sampling operations."""

from collections.abc import Sequence

import numpy as np
from onnx9000.converters.frontend.tensor import Tensor as FrontendTensor
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import AttributeValue, Tensor


def tensor(
    data: Sequence[float | int | bool] | np.ndarray,
    dtype: DType | None = None,
) -> FrontendTensor:
    """Construct a frontend Tensor from array-like data.

    Args:
        data: Input sequence or NumPy array.
        dtype: Optional explicit data type.

    Returns:
        Constructed FrontendTensor instance.
    """
    if dtype is None:
        return FrontendTensor(data=data)
    return FrontendTensor(data=data, dtype=dtype)


def zeros(*shape: int, dtype: DType = DType.FLOAT32) -> FrontendTensor:
    """Create a tensor filled with zeros.

    Args:
        *shape: Dimension sizes of the output tensor.
        dtype: Desired data type.

    Returns:
        Zero-initialized FrontendTensor.
    """
    return FrontendTensor(shape=shape, dtype=dtype, data=np.zeros(shape, dtype=np.float32))


def ones(*shape: int, dtype: DType = DType.FLOAT32) -> FrontendTensor:
    """Create a tensor filled with ones.

    Args:
        *shape: Dimension sizes of the output tensor.
        dtype: Desired data type.

    Returns:
        One-initialized FrontendTensor.
    """
    return FrontendTensor(shape=shape, dtype=dtype, data=np.ones(shape, dtype=np.float32))


def randn(*shape: int, dtype: DType = DType.FLOAT32) -> FrontendTensor:
    """Create a tensor sampled from a standard normal distribution.

    Args:
        *shape: Dimension sizes of the output tensor.
        dtype: Desired data type.

    Returns:
        Randomly initialized FrontendTensor.
    """
    return FrontendTensor(
        shape=shape,
        dtype=dtype,
        data=np.random.randn(*shape).astype(np.float32),
    )


def rand(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply rand to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for rand.
    """
    from onnx9000.core.ops.torch_auto import rand as core_rand

    return core_rand(input_tensor, *args, **kwargs)


def arange(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply arange to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for arange.
    """
    from onnx9000.core.ops.torch_auto import arange as core_arange

    return core_arange(input_tensor, *args, **kwargs)


def as_tensor(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply as_tensor to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for as_tensor.
    """
    from onnx9000.core.ops.torch_auto import as_tensor as core_as_tensor

    return core_as_tensor(input_tensor, *args, **kwargs)


def asarray(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply asarray to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for asarray.
    """
    from onnx9000.core.ops.torch_auto import asarray as core_asarray

    return core_asarray(input_tensor, *args, **kwargs)


def bartlett_window(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply bartlett_window to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for bartlett_window.
    """
    from onnx9000.core.ops.torch_auto import bartlett_window as core_bartlett_window

    return core_bartlett_window(input_tensor, *args, **kwargs)


def binomial(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply binomial to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for binomial.
    """
    from onnx9000.core.ops.torch_auto import binomial as core_binomial

    return core_binomial(input_tensor, *args, **kwargs)


def empty(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply empty to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for empty.
    """
    from onnx9000.core.ops.torch_auto import empty as core_empty

    return core_empty(input_tensor, *args, **kwargs)


def empty_like(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply empty_like to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for empty_like.
    """
    from onnx9000.core.ops.torch_auto import empty_like as core_empty_like

    return core_empty_like(input_tensor, *args, **kwargs)


def empty_permuted(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply empty_permuted to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for empty_permuted.
    """
    from onnx9000.core.ops.torch_auto import empty_permuted as core_empty_permuted

    return core_empty_permuted(input_tensor, *args, **kwargs)


def empty_quantized(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply empty_quantized to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for empty_quantized.
    """
    from onnx9000.core.ops.torch_auto import empty_quantized as core_empty_quantized

    return core_empty_quantized(input_tensor, *args, **kwargs)


def empty_strided(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply empty_strided to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for empty_strided.
    """
    from onnx9000.core.ops.torch_auto import empty_strided as core_empty_strided

    return core_empty_strided(input_tensor, *args, **kwargs)


def eye(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply eye to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for eye.
    """
    from onnx9000.core.ops.torch_auto import eye as core_eye

    return core_eye(input_tensor, *args, **kwargs)


def fill(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply fill to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for fill.
    """
    from onnx9000.core.ops.torch_auto import fill as core_fill

    return core_fill(input_tensor, *args, **kwargs)


def fill_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply fill_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for fill_.
    """
    from onnx9000.core.ops.torch_auto import fill_ as core_fill_

    return core_fill_(input_tensor, *args, **kwargs)


def from_file(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply from_file to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for from_file.
    """
    from onnx9000.core.ops.torch_auto import from_file as core_from_file

    return core_from_file(input_tensor, *args, **kwargs)


def from_numpy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply from_numpy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for from_numpy.
    """
    from onnx9000.core.ops.torch_auto import from_numpy as core_from_numpy

    return core_from_numpy(input_tensor, *args, **kwargs)


def frombuffer(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply frombuffer to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for frombuffer.
    """
    from onnx9000.core.ops.torch_auto import frombuffer as core_frombuffer

    return core_frombuffer(input_tensor, *args, **kwargs)


def full(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply full to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for full.
    """
    from onnx9000.core.ops.torch_auto import full as core_full

    return core_full(input_tensor, *args, **kwargs)


def full_like(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply full_like to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for full_like.
    """
    from onnx9000.core.ops.torch_auto import full_like as core_full_like

    return core_full_like(input_tensor, *args, **kwargs)


def hamming_window(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply hamming_window to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for hamming_window.
    """
    from onnx9000.core.ops.torch_auto import hamming_window as core_hamming_window

    return core_hamming_window(input_tensor, *args, **kwargs)


def hann_window(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply hann_window to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for hann_window.
    """
    from onnx9000.core.ops.torch_auto import hann_window as core_hann_window

    return core_hann_window(input_tensor, *args, **kwargs)


def kaiser_window(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply kaiser_window to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for kaiser_window.
    """
    from onnx9000.core.ops.torch_auto import kaiser_window as core_kaiser_window

    return core_kaiser_window(input_tensor, *args, **kwargs)


def linspace(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply linspace to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for linspace.
    """
    from onnx9000.core.ops.torch_auto import linspace as core_linspace

    return core_linspace(input_tensor, *args, **kwargs)


def logspace(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply logspace to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for logspace.
    """
    from onnx9000.core.ops.torch_auto import logspace as core_logspace

    return core_logspace(input_tensor, *args, **kwargs)


def normal(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply normal to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for normal.
    """
    from onnx9000.core.ops.torch_auto import normal as core_normal

    return core_normal(input_tensor, *args, **kwargs)


def ones_like(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply ones_like to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for ones_like.
    """
    from onnx9000.core.ops.torch_auto import ones_like as core_ones_like

    return core_ones_like(input_tensor, *args, **kwargs)


def poisson(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply poisson to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for poisson.
    """
    from onnx9000.core.ops.torch_auto import poisson as core_poisson

    return core_poisson(input_tensor, *args, **kwargs)


def rand_like(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply rand_like to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for rand_like.
    """
    from onnx9000.core.ops.torch_auto import rand_like as core_rand_like

    return core_rand_like(input_tensor, *args, **kwargs)


def randint(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply randint to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for randint.
    """
    from onnx9000.core.ops.torch_auto import randint as core_randint

    return core_randint(input_tensor, *args, **kwargs)


def randint_like(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply randint_like to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for randint_like.
    """
    from onnx9000.core.ops.torch_auto import randint_like as core_randint_like

    return core_randint_like(input_tensor, *args, **kwargs)


def randn_like(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply randn_like to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for randn_like.
    """
    from onnx9000.core.ops.torch_auto import randn_like as core_randn_like

    return core_randn_like(input_tensor, *args, **kwargs)


def randperm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply randperm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for randperm.
    """
    from onnx9000.core.ops.torch_auto import randperm as core_randperm

    return core_randperm(input_tensor, *args, **kwargs)


def range(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply range to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for range.
    """
    from onnx9000.core.ops.torch_auto import range as core_range

    return core_range(input_tensor, *args, **kwargs)


def scalar_tensor(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply scalar_tensor to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for scalar_tensor.
    """
    from onnx9000.core.ops.torch_auto import scalar_tensor as core_scalar_tensor

    return core_scalar_tensor(input_tensor, *args, **kwargs)


def sparse_bsc_tensor(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply sparse_bsc_tensor to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for sparse_bsc_tensor.
    """
    from onnx9000.core.ops.torch_auto import sparse_bsc_tensor as core_sparse_bsc_tensor

    return core_sparse_bsc_tensor(input_tensor, *args, **kwargs)


def sparse_bsr_tensor(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply sparse_bsr_tensor to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for sparse_bsr_tensor.
    """
    from onnx9000.core.ops.torch_auto import sparse_bsr_tensor as core_sparse_bsr_tensor

    return core_sparse_bsr_tensor(input_tensor, *args, **kwargs)


def sparse_compressed_tensor(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply sparse_compressed_tensor to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for sparse_compressed_tensor.
    """
    from onnx9000.core.ops.torch_auto import (
        sparse_compressed_tensor as core_sparse_compressed_tensor,
    )

    return core_sparse_compressed_tensor(input_tensor, *args, **kwargs)


def sparse_coo_tensor(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply sparse_coo_tensor to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for sparse_coo_tensor.
    """
    from onnx9000.core.ops.torch_auto import sparse_coo_tensor as core_sparse_coo_tensor

    return core_sparse_coo_tensor(input_tensor, *args, **kwargs)


def sparse_csc_tensor(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply sparse_csc_tensor to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for sparse_csc_tensor.
    """
    from onnx9000.core.ops.torch_auto import sparse_csc_tensor as core_sparse_csc_tensor

    return core_sparse_csc_tensor(input_tensor, *args, **kwargs)


def sparse_csr_tensor(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply sparse_csr_tensor to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for sparse_csr_tensor.
    """
    from onnx9000.core.ops.torch_auto import sparse_csr_tensor as core_sparse_csr_tensor

    return core_sparse_csr_tensor(input_tensor, *args, **kwargs)


def tril_indices(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply tril_indices to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for tril_indices.
    """
    from onnx9000.core.ops.torch_auto import tril_indices as core_tril_indices

    return core_tril_indices(input_tensor, *args, **kwargs)


def triu_indices(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply triu_indices to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for triu_indices.
    """
    from onnx9000.core.ops.torch_auto import triu_indices as core_triu_indices

    return core_triu_indices(input_tensor, *args, **kwargs)


def vander(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply vander to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for vander.
    """
    from onnx9000.core.ops.torch_auto import vander as core_vander

    return core_vander(input_tensor, *args, **kwargs)


def zero_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply zero_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for zero_.
    """
    from onnx9000.core.ops.torch_auto import zero_ as core_zero_

    return core_zero_(input_tensor, *args, **kwargs)


def zeros_like(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply zeros_like to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for zeros_like.
    """
    from onnx9000.core.ops.torch_auto import zeros_like as core_zeros_like

    return core_zeros_like(input_tensor, *args, **kwargs)
