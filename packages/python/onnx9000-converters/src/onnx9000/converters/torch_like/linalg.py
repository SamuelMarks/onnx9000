"""PyTorch-compatible linear algebra and spectral operations."""

from onnx9000.core.ir import AttributeValue, Tensor


def matmul(input_tensor: Tensor, other: Tensor) -> Tensor:
    """Compute matrix product of two tensors.

    Args:
        input_tensor: Left-hand operand tensor.
        other: Right-hand operand tensor.

    Returns:
        Symbolic output Tensor for MatMul.
    """
    from onnx9000.core.ops import matmul as core_matmul

    return core_matmul(input_tensor, other)


def lobpcg(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply lobpcg to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for lobpcg.
    """
    from onnx9000.core.ops.torch_auto import lobpcg as core_lobpcg

    return core_lobpcg(input_tensor, *args, **kwargs)


def addbmm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply addbmm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for addbmm.
    """
    from onnx9000.core.ops.torch_auto import addbmm as core_addbmm

    return core_addbmm(input_tensor, *args, **kwargs)


def addmm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply addmm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for addmm.
    """
    from onnx9000.core.ops.torch_auto import addmm as core_addmm

    return core_addmm(input_tensor, *args, **kwargs)


def addmv(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply addmv to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for addmv.
    """
    from onnx9000.core.ops.torch_auto import addmv as core_addmv

    return core_addmv(input_tensor, *args, **kwargs)


def addmv_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply addmv_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for addmv_.
    """
    from onnx9000.core.ops.torch_auto import addmv_ as core_addmv_

    return core_addmv_(input_tensor, *args, **kwargs)


def addr(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply addr to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for addr.
    """
    from onnx9000.core.ops.torch_auto import addr as core_addr

    return core_addr(input_tensor, *args, **kwargs)


def adjoint(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply adjoint to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for adjoint.
    """
    from onnx9000.core.ops.torch_auto import adjoint as core_adjoint

    return core_adjoint(input_tensor, *args, **kwargs)


def baddbmm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply baddbmm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for baddbmm.
    """
    from onnx9000.core.ops.torch_auto import baddbmm as core_baddbmm

    return core_baddbmm(input_tensor, *args, **kwargs)


def bmm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply bmm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for bmm.
    """
    from onnx9000.core.ops.torch_auto import bmm as core_bmm

    return core_bmm(input_tensor, *args, **kwargs)


def chain_matmul(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply chain_matmul to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for chain_matmul.
    """
    from onnx9000.core.ops.torch_auto import chain_matmul as core_chain_matmul

    return core_chain_matmul(input_tensor, *args, **kwargs)


def cholesky(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cholesky to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cholesky.
    """
    from onnx9000.core.ops.torch_auto import cholesky as core_cholesky

    return core_cholesky(input_tensor, *args, **kwargs)


def cholesky_inverse(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cholesky_inverse to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cholesky_inverse.
    """
    from onnx9000.core.ops.torch_auto import cholesky_inverse as core_cholesky_inverse

    return core_cholesky_inverse(input_tensor, *args, **kwargs)


def cholesky_solve(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cholesky_solve to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cholesky_solve.
    """
    from onnx9000.core.ops.torch_auto import cholesky_solve as core_cholesky_solve

    return core_cholesky_solve(input_tensor, *args, **kwargs)


def corrcoef(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply corrcoef to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for corrcoef.
    """
    from onnx9000.core.ops.torch_auto import corrcoef as core_corrcoef

    return core_corrcoef(input_tensor, *args, **kwargs)


def cov(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cov to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cov.
    """
    from onnx9000.core.ops.torch_auto import cov as core_cov

    return core_cov(input_tensor, *args, **kwargs)


def cross(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cross to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cross.
    """
    from onnx9000.core.ops.torch_auto import cross as core_cross

    return core_cross(input_tensor, *args, **kwargs)


def diag(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply diag to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for diag.
    """
    from onnx9000.core.ops.torch_auto import diag as core_diag

    return core_diag(input_tensor, *args, **kwargs)


def diag_embed(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply diag_embed to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for diag_embed.
    """
    from onnx9000.core.ops.torch_auto import diag_embed as core_diag_embed

    return core_diag_embed(input_tensor, *args, **kwargs)


def diagflat(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply diagflat to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for diagflat.
    """
    from onnx9000.core.ops.torch_auto import diagflat as core_diagflat

    return core_diagflat(input_tensor, *args, **kwargs)


def diagonal(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply diagonal to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for diagonal.
    """
    from onnx9000.core.ops.torch_auto import diagonal as core_diagonal

    return core_diagonal(input_tensor, *args, **kwargs)


def diagonal_copy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply diagonal_copy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for diagonal_copy.
    """
    from onnx9000.core.ops.torch_auto import diagonal_copy as core_diagonal_copy

    return core_diagonal_copy(input_tensor, *args, **kwargs)


def diagonal_scatter(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply diagonal_scatter to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for diagonal_scatter.
    """
    from onnx9000.core.ops.torch_auto import diagonal_scatter as core_diagonal_scatter

    return core_diagonal_scatter(input_tensor, *args, **kwargs)


def dot(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply dot to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for dot.
    """
    from onnx9000.core.ops.torch_auto import dot as core_dot

    return core_dot(input_tensor, *args, **kwargs)


def dsmm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply dsmm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for dsmm.
    """
    from onnx9000.core.ops.torch_auto import dsmm as core_dsmm

    return core_dsmm(input_tensor, *args, **kwargs)


def frobenius_norm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply frobenius_norm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for frobenius_norm.
    """
    from onnx9000.core.ops.torch_auto import frobenius_norm as core_frobenius_norm

    return core_frobenius_norm(input_tensor, *args, **kwargs)


def geqrf(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply geqrf to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for geqrf.
    """
    from onnx9000.core.ops.torch_auto import geqrf as core_geqrf

    return core_geqrf(input_tensor, *args, **kwargs)


def ger(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply ger to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for ger.
    """
    from onnx9000.core.ops.torch_auto import ger as core_ger

    return core_ger(input_tensor, *args, **kwargs)


def hsmm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply hsmm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for hsmm.
    """
    from onnx9000.core.ops.torch_auto import hsmm as core_hsmm

    return core_hsmm(input_tensor, *args, **kwargs)


def hspmm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply hspmm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for hspmm.
    """
    from onnx9000.core.ops.torch_auto import hspmm as core_hspmm

    return core_hspmm(input_tensor, *args, **kwargs)


def inner(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply inner to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for inner.
    """
    from onnx9000.core.ops.torch_auto import inner as core_inner

    return core_inner(input_tensor, *args, **kwargs)


def inverse(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply inverse to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for inverse.
    """
    from onnx9000.core.ops.torch_auto import inverse as core_inverse

    return core_inverse(input_tensor, *args, **kwargs)


def istft(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply istft to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for istft.
    """
    from onnx9000.core.ops.torch_auto import istft as core_istft

    return core_istft(input_tensor, *args, **kwargs)


def kron(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply kron to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for kron.
    """
    from onnx9000.core.ops.torch_auto import kron as core_kron

    return core_kron(input_tensor, *args, **kwargs)


def logdet(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply logdet to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for logdet.
    """
    from onnx9000.core.ops.torch_auto import logdet as core_logdet

    return core_logdet(input_tensor, *args, **kwargs)


def lu_solve(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply lu_solve to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for lu_solve.
    """
    from onnx9000.core.ops.torch_auto import lu_solve as core_lu_solve

    return core_lu_solve(input_tensor, *args, **kwargs)


def lu_unpack(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply lu_unpack to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for lu_unpack.
    """
    from onnx9000.core.ops.torch_auto import lu_unpack as core_lu_unpack

    return core_lu_unpack(input_tensor, *args, **kwargs)


def matrix_exp(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply matrix_exp to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for matrix_exp.
    """
    from onnx9000.core.ops.torch_auto import matrix_exp as core_matrix_exp

    return core_matrix_exp(input_tensor, *args, **kwargs)


def matrix_power(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply matrix_power to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for matrix_power.
    """
    from onnx9000.core.ops.torch_auto import matrix_power as core_matrix_power

    return core_matrix_power(input_tensor, *args, **kwargs)


def mm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply mm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for mm.
    """
    from onnx9000.core.ops.torch_auto import mm as core_mm

    return core_mm(input_tensor, *args, **kwargs)


def mv(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply mv to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for mv.
    """
    from onnx9000.core.ops.torch_auto import mv as core_mv

    return core_mv(input_tensor, *args, **kwargs)


def norm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply norm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for norm.
    """
    from onnx9000.core.ops.torch_auto import norm as core_norm

    return core_norm(input_tensor, *args, **kwargs)


def nuclear_norm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply nuclear_norm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for nuclear_norm.
    """
    from onnx9000.core.ops.torch_auto import nuclear_norm as core_nuclear_norm

    return core_nuclear_norm(input_tensor, *args, **kwargs)


def orgqr(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply orgqr to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for orgqr.
    """
    from onnx9000.core.ops.torch_auto import orgqr as core_orgqr

    return core_orgqr(input_tensor, *args, **kwargs)


def ormqr(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply ormqr to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for ormqr.
    """
    from onnx9000.core.ops.torch_auto import ormqr as core_ormqr

    return core_ormqr(input_tensor, *args, **kwargs)


def outer(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply outer to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for outer.
    """
    from onnx9000.core.ops.torch_auto import outer as core_outer

    return core_outer(input_tensor, *args, **kwargs)


def pinverse(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply pinverse to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for pinverse.
    """
    from onnx9000.core.ops.torch_auto import pinverse as core_pinverse

    return core_pinverse(input_tensor, *args, **kwargs)


def qr(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply qr to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for qr.
    """
    from onnx9000.core.ops.torch_auto import qr as core_qr

    return core_qr(input_tensor, *args, **kwargs)


def saddmm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply saddmm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for saddmm.
    """
    from onnx9000.core.ops.torch_auto import saddmm as core_saddmm

    return core_saddmm(input_tensor, *args, **kwargs)


def slogdet(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply slogdet to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for slogdet.
    """
    from onnx9000.core.ops.torch_auto import slogdet as core_slogdet

    return core_slogdet(input_tensor, *args, **kwargs)


def smm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply smm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for smm.
    """
    from onnx9000.core.ops.torch_auto import smm as core_smm

    return core_smm(input_tensor, *args, **kwargs)


def spmm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply spmm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for spmm.
    """
    from onnx9000.core.ops.torch_auto import spmm as core_spmm

    return core_spmm(input_tensor, *args, **kwargs)


def sspaddmm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply sspaddmm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for sspaddmm.
    """
    from onnx9000.core.ops.torch_auto import sspaddmm as core_sspaddmm

    return core_sspaddmm(input_tensor, *args, **kwargs)


def stft(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply stft to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for stft.
    """
    from onnx9000.core.ops.torch_auto import stft as core_stft

    return core_stft(input_tensor, *args, **kwargs)


def svd(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply svd to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for svd.
    """
    from onnx9000.core.ops.torch_auto import svd as core_svd

    return core_svd(input_tensor, *args, **kwargs)


def tensordot(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply tensordot to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for tensordot.
    """
    from onnx9000.core.ops.torch_auto import tensordot as core_tensordot

    return core_tensordot(input_tensor, *args, **kwargs)


def triangular_solve(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply triangular_solve to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for triangular_solve.
    """
    from onnx9000.core.ops.torch_auto import triangular_solve as core_triangular_solve

    return core_triangular_solve(input_tensor, *args, **kwargs)


def tril(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply tril to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for tril.
    """
    from onnx9000.core.ops.torch_auto import tril as core_tril

    return core_tril(input_tensor, *args, **kwargs)


def triu(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply triu to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for triu.
    """
    from onnx9000.core.ops.torch_auto import triu as core_triu

    return core_triu(input_tensor, *args, **kwargs)


def vdot(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply vdot to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for vdot.
    """
    from onnx9000.core.ops.torch_auto import vdot as core_vdot

    return core_vdot(input_tensor, *args, **kwargs)
