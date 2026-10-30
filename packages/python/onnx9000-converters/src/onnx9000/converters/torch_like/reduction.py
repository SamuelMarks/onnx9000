"""PyTorch-compatible reduction and statistical operations."""

from onnx9000.core.ir import AttributeValue, Tensor


def all(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply all to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for all.
    """
    from onnx9000.core.ops.torch_auto import all as core_all

    return core_all(input_tensor, *args, **kwargs)


def allclose(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply allclose to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for allclose.
    """
    from onnx9000.core.ops.torch_auto import allclose as core_allclose

    return core_allclose(input_tensor, *args, **kwargs)


def amax(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply amax to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for amax.
    """
    from onnx9000.core.ops.torch_auto import amax as core_amax

    return core_amax(input_tensor, *args, **kwargs)


def amin(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply amin to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for amin.
    """
    from onnx9000.core.ops.torch_auto import amin as core_amin

    return core_amin(input_tensor, *args, **kwargs)


def aminmax(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply aminmax to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for aminmax.
    """
    from onnx9000.core.ops.torch_auto import aminmax as core_aminmax

    return core_aminmax(input_tensor, *args, **kwargs)


def any(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply any to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for any.
    """
    from onnx9000.core.ops.torch_auto import any as core_any

    return core_any(input_tensor, *args, **kwargs)


def bincount(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply bincount to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for bincount.
    """
    from onnx9000.core.ops.torch_auto import bincount as core_bincount

    return core_bincount(input_tensor, *args, **kwargs)


def count_nonzero(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply count_nonzero to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for count_nonzero.
    """
    from onnx9000.core.ops.torch_auto import count_nonzero as core_count_nonzero

    return core_count_nonzero(input_tensor, *args, **kwargs)


def cummax(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cummax to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cummax.
    """
    from onnx9000.core.ops.torch_auto import cummax as core_cummax

    return core_cummax(input_tensor, *args, **kwargs)


def cummin(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cummin to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cummin.
    """
    from onnx9000.core.ops.torch_auto import cummin as core_cummin

    return core_cummin(input_tensor, *args, **kwargs)


def cumprod(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cumprod to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cumprod.
    """
    from onnx9000.core.ops.torch_auto import cumprod as core_cumprod

    return core_cumprod(input_tensor, *args, **kwargs)


def cumulative_trapezoid(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cumulative_trapezoid to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cumulative_trapezoid.
    """
    from onnx9000.core.ops.torch_auto import cumulative_trapezoid as core_cumulative_trapezoid

    return core_cumulative_trapezoid(input_tensor, *args, **kwargs)


def dist(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply dist to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for dist.
    """
    from onnx9000.core.ops.torch_auto import dist as core_dist

    return core_dist(input_tensor, *args, **kwargs)


def histc(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply histc to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for histc.
    """
    from onnx9000.core.ops.torch_auto import histc as core_histc

    return core_histc(input_tensor, *args, **kwargs)


def histogram(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply histogram to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for histogram.
    """
    from onnx9000.core.ops.torch_auto import histogram as core_histogram

    return core_histogram(input_tensor, *args, **kwargs)


def histogramdd(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply histogramdd to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for histogramdd.
    """
    from onnx9000.core.ops.torch_auto import histogramdd as core_histogramdd

    return core_histogramdd(input_tensor, *args, **kwargs)


def kthvalue(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply kthvalue to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for kthvalue.
    """
    from onnx9000.core.ops.torch_auto import kthvalue as core_kthvalue

    return core_kthvalue(input_tensor, *args, **kwargs)


def logcumsumexp(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply logcumsumexp to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for logcumsumexp.
    """
    from onnx9000.core.ops.torch_auto import logcumsumexp as core_logcumsumexp

    return core_logcumsumexp(input_tensor, *args, **kwargs)


def logsumexp(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply logsumexp to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for logsumexp.
    """
    from onnx9000.core.ops.torch_auto import logsumexp as core_logsumexp

    return core_logsumexp(input_tensor, *args, **kwargs)


def median(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply median to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for median.
    """
    from onnx9000.core.ops.torch_auto import median as core_median

    return core_median(input_tensor, *args, **kwargs)


def mode(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply mode to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for mode.
    """
    from onnx9000.core.ops.torch_auto import mode as core_mode

    return core_mode(input_tensor, *args, **kwargs)


def nanmean(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply nanmean to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for nanmean.
    """
    from onnx9000.core.ops.torch_auto import nanmean as core_nanmean

    return core_nanmean(input_tensor, *args, **kwargs)


def nanmedian(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply nanmedian to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for nanmedian.
    """
    from onnx9000.core.ops.torch_auto import nanmedian as core_nanmedian

    return core_nanmedian(input_tensor, *args, **kwargs)


def nanquantile(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply nanquantile to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for nanquantile.
    """
    from onnx9000.core.ops.torch_auto import nanquantile as core_nanquantile

    return core_nanquantile(input_tensor, *args, **kwargs)


def nansum(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply nansum to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for nansum.
    """
    from onnx9000.core.ops.torch_auto import nansum as core_nansum

    return core_nansum(input_tensor, *args, **kwargs)


def norm_except_dim(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply norm_except_dim to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for norm_except_dim.
    """
    from onnx9000.core.ops.torch_auto import norm_except_dim as core_norm_except_dim

    return core_norm_except_dim(input_tensor, *args, **kwargs)


def numel(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply numel to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for numel.
    """
    from onnx9000.core.ops.torch_auto import numel as core_numel

    return core_numel(input_tensor, *args, **kwargs)


def prod(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply prod to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for prod.
    """
    from onnx9000.core.ops.torch_auto import prod as core_prod

    return core_prod(input_tensor, *args, **kwargs)


def quantile(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply quantile to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for quantile.
    """
    from onnx9000.core.ops.torch_auto import quantile as core_quantile

    return core_quantile(input_tensor, *args, **kwargs)


def std(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply std to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for std.
    """
    from onnx9000.core.ops.torch_auto import std as core_std

    return core_std(input_tensor, *args, **kwargs)


def std_mean(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply std_mean to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for std_mean.
    """
    from onnx9000.core.ops.torch_auto import std_mean as core_std_mean

    return core_std_mean(input_tensor, *args, **kwargs)


def trapezoid(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply trapezoid to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for trapezoid.
    """
    from onnx9000.core.ops.torch_auto import trapezoid as core_trapezoid

    return core_trapezoid(input_tensor, *args, **kwargs)


def trapz(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply trapz to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for trapz.
    """
    from onnx9000.core.ops.torch_auto import trapz as core_trapz

    return core_trapz(input_tensor, *args, **kwargs)


def unique_consecutive(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply unique_consecutive to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for unique_consecutive.
    """
    from onnx9000.core.ops.torch_auto import unique_consecutive as core_unique_consecutive

    return core_unique_consecutive(input_tensor, *args, **kwargs)


def var(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply var to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for var.
    """
    from onnx9000.core.ops.torch_auto import var as core_var

    return core_var(input_tensor, *args, **kwargs)


def var_mean(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply var_mean to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for var_mean.
    """
    from onnx9000.core.ops.torch_auto import var_mean as core_var_mean

    return core_var_mean(input_tensor, *args, **kwargs)
