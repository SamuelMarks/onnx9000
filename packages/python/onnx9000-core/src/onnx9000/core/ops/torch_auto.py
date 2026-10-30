"""PyTorch ATen operator delegation helpers for symbolic graph recording."""

from onnx9000.core.ir import AttributeValue, Tensor
from onnx9000.core.ops.base import record_op


def chunk(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Chunk operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Chunk output.
    """
    return record_op("Chunk", [x, *args], kwargs)


def lobpcg(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Lobpcg operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Lobpcg output.
    """
    return record_op("Lobpcg", [x, *args], kwargs)


def rand(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Rand operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Rand output.
    """
    return record_op("Rand", [x, *args], kwargs)


def split(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Split operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Split output.
    """
    return record_op("Split", [x, *args], kwargs)


def stack(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Stack operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Stack output.
    """
    return record_op("Stack", [x, *args], kwargs)


def unravel_index(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Unravel_index operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Unravel_index output.
    """
    return record_op("Unravel_index", [x, *args], kwargs)


def abs_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Abs_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Abs_ output.
    """
    return record_op("Abs_", [x, *args], kwargs)


def absolute(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Absolute operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Absolute output.
    """
    return record_op("Absolute", [x, *args], kwargs)


def acos_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Acos_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Acos_ output.
    """
    return record_op("Acos_", [x, *args], kwargs)


def acosh_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Acosh_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Acosh_ output.
    """
    return record_op("Acosh_", [x, *args], kwargs)


def adaptive_avg_pool1d(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Adaptive_avg_pool1d operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Adaptive_avg_pool1d output.
    """
    return record_op("Adaptive_avg_pool1d", [x, *args], kwargs)


def adaptive_max_pool1d(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Adaptive_max_pool1d operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Adaptive_max_pool1d output.
    """
    return record_op("Adaptive_max_pool1d", [x, *args], kwargs)


def addbmm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Addbmm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Addbmm output.
    """
    return record_op("Addbmm", [x, *args], kwargs)


def addcdiv(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Addcdiv operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Addcdiv output.
    """
    return record_op("Addcdiv", [x, *args], kwargs)


def addcmul(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Addcmul operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Addcmul output.
    """
    return record_op("Addcmul", [x, *args], kwargs)


def addmm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Addmm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Addmm output.
    """
    return record_op("Addmm", [x, *args], kwargs)


def addmv(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Addmv operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Addmv output.
    """
    return record_op("Addmv", [x, *args], kwargs)


def addmv_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Addmv_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Addmv_ output.
    """
    return record_op("Addmv_", [x, *args], kwargs)


def addr(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Addr operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Addr output.
    """
    return record_op("Addr", [x, *args], kwargs)


def adjoint(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Adjoint operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Adjoint output.
    """
    return record_op("Adjoint", [x, *args], kwargs)


def affine_grid_generator(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Affine_grid_generator operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Affine_grid_generator output.
    """
    return record_op("Affine_grid_generator", [x, *args], kwargs)


def alias_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Alias_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Alias_copy output.
    """
    return record_op("Alias_copy", [x, *args], kwargs)


def align_tensors(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Align_tensors operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Align_tensors output.
    """
    return record_op("Align_tensors", [x, *args], kwargs)


def all(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the All operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the All output.
    """
    return record_op("All", [x, *args], kwargs)


def allclose(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Allclose operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Allclose output.
    """
    return record_op("Allclose", [x, *args], kwargs)


def alpha_dropout(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Alpha_dropout operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Alpha_dropout output.
    """
    return record_op("Alpha_dropout", [x, *args], kwargs)


def alpha_dropout_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Alpha_dropout_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Alpha_dropout_ output.
    """
    return record_op("Alpha_dropout_", [x, *args], kwargs)


def amax(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Amax operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Amax output.
    """
    return record_op("Amax", [x, *args], kwargs)


def amin(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Amin operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Amin output.
    """
    return record_op("Amin", [x, *args], kwargs)


def aminmax(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Aminmax operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Aminmax output.
    """
    return record_op("Aminmax", [x, *args], kwargs)


def angle(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Angle operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Angle output.
    """
    return record_op("Angle", [x, *args], kwargs)


def any(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Any operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Any output.
    """
    return record_op("Any", [x, *args], kwargs)


def arange(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Arange operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Arange output.
    """
    return record_op("Arange", [x, *args], kwargs)


def arccos(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Arccos operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Arccos output.
    """
    return record_op("Arccos", [x, *args], kwargs)


def arccos_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Arccos_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Arccos_ output.
    """
    return record_op("Arccos_", [x, *args], kwargs)


def arccosh(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Arccosh operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Arccosh output.
    """
    return record_op("Arccosh", [x, *args], kwargs)


def arccosh_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Arccosh_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Arccosh_ output.
    """
    return record_op("Arccosh_", [x, *args], kwargs)


def arcsin(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Arcsin operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Arcsin output.
    """
    return record_op("Arcsin", [x, *args], kwargs)


def arcsin_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Arcsin_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Arcsin_ output.
    """
    return record_op("Arcsin_", [x, *args], kwargs)


def arcsinh(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Arcsinh operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Arcsinh output.
    """
    return record_op("Arcsinh", [x, *args], kwargs)


def arcsinh_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Arcsinh_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Arcsinh_ output.
    """
    return record_op("Arcsinh_", [x, *args], kwargs)


def arctan(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Arctan operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Arctan output.
    """
    return record_op("Arctan", [x, *args], kwargs)


def arctan2(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Arctan2 operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Arctan2 output.
    """
    return record_op("Arctan2", [x, *args], kwargs)


def arctan_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Arctan_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Arctan_ output.
    """
    return record_op("Arctan_", [x, *args], kwargs)


def arctanh(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Arctanh operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Arctanh output.
    """
    return record_op("Arctanh", [x, *args], kwargs)


def arctanh_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Arctanh_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Arctanh_ output.
    """
    return record_op("Arctanh_", [x, *args], kwargs)


def argsort(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Argsort operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Argsort output.
    """
    return record_op("Argsort", [x, *args], kwargs)


def argwhere(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Argwhere operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Argwhere output.
    """
    return record_op("Argwhere", [x, *args], kwargs)


def as_strided(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the As_strided operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the As_strided output.
    """
    return record_op("As_strided", [x, *args], kwargs)


def as_strided_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the As_strided_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the As_strided_ output.
    """
    return record_op("As_strided_", [x, *args], kwargs)


def as_strided_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the As_strided_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the As_strided_copy output.
    """
    return record_op("As_strided_copy", [x, *args], kwargs)


def as_strided_scatter(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the As_strided_scatter operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the As_strided_scatter output.
    """
    return record_op("As_strided_scatter", [x, *args], kwargs)


def as_tensor(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the As_tensor operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the As_tensor output.
    """
    return record_op("As_tensor", [x, *args], kwargs)


def asarray(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Asarray operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Asarray output.
    """
    return record_op("Asarray", [x, *args], kwargs)


def asin_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Asin_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Asin_ output.
    """
    return record_op("Asin_", [x, *args], kwargs)


def asinh_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Asinh_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Asinh_ output.
    """
    return record_op("Asinh_", [x, *args], kwargs)


def atan2(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Atan2 operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Atan2 output.
    """
    return record_op("Atan2", [x, *args], kwargs)


def atan_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Atan_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Atan_ output.
    """
    return record_op("Atan_", [x, *args], kwargs)


def atanh_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Atanh_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Atanh_ output.
    """
    return record_op("Atanh_", [x, *args], kwargs)


def atleast_1d(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Atleast_1d operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Atleast_1d output.
    """
    return record_op("Atleast_1d", [x, *args], kwargs)


def atleast_2d(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Atleast_2d operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Atleast_2d output.
    """
    return record_op("Atleast_2d", [x, *args], kwargs)


def atleast_3d(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Atleast_3d operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Atleast_3d output.
    """
    return record_op("Atleast_3d", [x, *args], kwargs)


def avg_pool1d(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Avg_pool1d operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Avg_pool1d output.
    """
    return record_op("Avg_pool1d", [x, *args], kwargs)


def baddbmm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Baddbmm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Baddbmm output.
    """
    return record_op("Baddbmm", [x, *args], kwargs)


def bartlett_window(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Bartlett_window operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Bartlett_window output.
    """
    return record_op("Bartlett_window", [x, *args], kwargs)


def batch_norm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Batch_norm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Batch_norm output.
    """
    return record_op("Batch_norm", [x, *args], kwargs)


def batch_norm_backward_elemt(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Batch_norm_backward_elemt operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Batch_norm_backward_elemt output.
    """
    return record_op("Batch_norm_backward_elemt", [x, *args], kwargs)


def batch_norm_backward_reduce(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Batch_norm_backward_reduce operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Batch_norm_backward_reduce output.
    """
    return record_op("Batch_norm_backward_reduce", [x, *args], kwargs)


def batch_norm_elemt(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Batch_norm_elemt operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Batch_norm_elemt output.
    """
    return record_op("Batch_norm_elemt", [x, *args], kwargs)


def batch_norm_gather_stats(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Batch_norm_gather_stats operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Batch_norm_gather_stats output.
    """
    return record_op("Batch_norm_gather_stats", [x, *args], kwargs)


def batch_norm_gather_stats_with_counts(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Batch_norm_gather_stats_with_counts operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Batch_norm_gather_stats_with_counts output.
    """
    return record_op("Batch_norm_gather_stats_with_counts", [x, *args], kwargs)


def batch_norm_stats(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Batch_norm_stats operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Batch_norm_stats output.
    """
    return record_op("Batch_norm_stats", [x, *args], kwargs)


def batch_norm_update_stats(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Batch_norm_update_stats operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Batch_norm_update_stats output.
    """
    return record_op("Batch_norm_update_stats", [x, *args], kwargs)


def bilinear(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Bilinear operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Bilinear output.
    """
    return record_op("Bilinear", [x, *args], kwargs)


def binary_cross_entropy_with_logits(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Binary_cross_entropy_with_logits operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Binary_cross_entropy_with_logits output.
    """
    return record_op("Binary_cross_entropy_with_logits", [x, *args], kwargs)


def bincount(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Bincount operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Bincount output.
    """
    return record_op("Bincount", [x, *args], kwargs)


def binomial(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Binomial operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Binomial output.
    """
    return record_op("Binomial", [x, *args], kwargs)


def bitwise_left_shift(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Bitwise_left_shift operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Bitwise_left_shift output.
    """
    return record_op("Bitwise_left_shift", [x, *args], kwargs)


def bitwise_right_shift(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Bitwise_right_shift operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Bitwise_right_shift output.
    """
    return record_op("Bitwise_right_shift", [x, *args], kwargs)


def block_diag(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Block_diag operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Block_diag output.
    """
    return record_op("Block_diag", [x, *args], kwargs)


def bmm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Bmm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Bmm output.
    """
    return record_op("Bmm", [x, *args], kwargs)


def broadcast_tensors(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Broadcast_tensors operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Broadcast_tensors output.
    """
    return record_op("Broadcast_tensors", [x, *args], kwargs)


def broadcast_to(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Broadcast_to operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Broadcast_to output.
    """
    return record_op("Broadcast_to", [x, *args], kwargs)


def bucketize(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Bucketize operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Bucketize output.
    """
    return record_op("Bucketize", [x, *args], kwargs)


def cartesian_prod(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Cartesian_prod operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Cartesian_prod output.
    """
    return record_op("Cartesian_prod", [x, *args], kwargs)


def cat(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Cat operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Cat output.
    """
    return record_op("Cat", [x, *args], kwargs)


def ccol_indices_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Ccol_indices_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Ccol_indices_copy output.
    """
    return record_op("Ccol_indices_copy", [x, *args], kwargs)


def cdist(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Cdist operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Cdist output.
    """
    return record_op("Cdist", [x, *args], kwargs)


def ceil_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Ceil_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Ceil_ output.
    """
    return record_op("Ceil_", [x, *args], kwargs)


def celu_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Celu_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Celu_ output.
    """
    return record_op("Celu_", [x, *args], kwargs)


def chain_matmul(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Chain_matmul operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Chain_matmul output.
    """
    return record_op("Chain_matmul", [x, *args], kwargs)


def channel_shuffle(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Channel_shuffle operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Channel_shuffle output.
    """
    return record_op("Channel_shuffle", [x, *args], kwargs)


def cholesky(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Cholesky operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Cholesky output.
    """
    return record_op("Cholesky", [x, *args], kwargs)


def cholesky_inverse(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Cholesky_inverse operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Cholesky_inverse output.
    """
    return record_op("Cholesky_inverse", [x, *args], kwargs)


def cholesky_solve(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Cholesky_solve operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Cholesky_solve output.
    """
    return record_op("Cholesky_solve", [x, *args], kwargs)


def choose_qparams_optimized(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Choose_qparams_optimized operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Choose_qparams_optimized output.
    """
    return record_op("Choose_qparams_optimized", [x, *args], kwargs)


def clamp(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Clamp operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Clamp output.
    """
    return record_op("Clamp", [x, *args], kwargs)


def clamp_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Clamp_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Clamp_ output.
    """
    return record_op("Clamp_", [x, *args], kwargs)


def clamp_max(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Clamp_max operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Clamp_max output.
    """
    return record_op("Clamp_max", [x, *args], kwargs)


def clamp_max_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Clamp_max_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Clamp_max_ output.
    """
    return record_op("Clamp_max_", [x, *args], kwargs)


def clamp_min(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Clamp_min operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Clamp_min output.
    """
    return record_op("Clamp_min", [x, *args], kwargs)


def clamp_min_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Clamp_min_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Clamp_min_ output.
    """
    return record_op("Clamp_min_", [x, *args], kwargs)


def clip_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Clip_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Clip_ output.
    """
    return record_op("Clip_", [x, *args], kwargs)


def clone(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Clone operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Clone output.
    """
    return record_op("Clone", [x, *args], kwargs)


def col_indices_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Col_indices_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Col_indices_copy output.
    """
    return record_op("Col_indices_copy", [x, *args], kwargs)


def column_stack(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Column_stack operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Column_stack output.
    """
    return record_op("Column_stack", [x, *args], kwargs)


def combinations(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Combinations operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Combinations output.
    """
    return record_op("Combinations", [x, *args], kwargs)


def complex(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Complex operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Complex output.
    """
    return record_op("Complex", [x, *args], kwargs)


def concatenate(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Concatenate operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Concatenate output.
    """
    return record_op("Concatenate", [x, *args], kwargs)


def conj(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Conj operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Conj output.
    """
    return record_op("Conj", [x, *args], kwargs)


def conj_physical(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Conj_physical operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Conj_physical output.
    """
    return record_op("Conj_physical", [x, *args], kwargs)


def conj_physical_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Conj_physical_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Conj_physical_ output.
    """
    return record_op("Conj_physical_", [x, *args], kwargs)


def constant_pad_nd(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Constant_pad_nd operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Constant_pad_nd output.
    """
    return record_op("Constant_pad_nd", [x, *args], kwargs)


def conv1d(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Conv1d operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Conv1d output.
    """
    return record_op("Conv1d", [x, *args], kwargs)


def conv2d(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Conv2d operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Conv2d output.
    """
    return record_op("Conv2d", [x, *args], kwargs)


def conv3d(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Conv3d operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Conv3d output.
    """
    return record_op("Conv3d", [x, *args], kwargs)


def conv_tbc(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Conv_tbc operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Conv_tbc output.
    """
    return record_op("Conv_tbc", [x, *args], kwargs)


def conv_transpose1d(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Conv_transpose1d operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Conv_transpose1d output.
    """
    return record_op("Conv_transpose1d", [x, *args], kwargs)


def conv_transpose2d(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Conv_transpose2d operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Conv_transpose2d output.
    """
    return record_op("Conv_transpose2d", [x, *args], kwargs)


def conv_transpose3d(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Conv_transpose3d operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Conv_transpose3d output.
    """
    return record_op("Conv_transpose3d", [x, *args], kwargs)


def convolution(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Convolution operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Convolution output.
    """
    return record_op("Convolution", [x, *args], kwargs)


def copysign(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Copysign operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Copysign output.
    """
    return record_op("Copysign", [x, *args], kwargs)


def corrcoef(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Corrcoef operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Corrcoef output.
    """
    return record_op("Corrcoef", [x, *args], kwargs)


def cos_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Cos_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Cos_ output.
    """
    return record_op("Cos_", [x, *args], kwargs)


def cosh_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Cosh_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Cosh_ output.
    """
    return record_op("Cosh_", [x, *args], kwargs)


def cosine_embedding_loss(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Cosine_embedding_loss operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Cosine_embedding_loss output.
    """
    return record_op("Cosine_embedding_loss", [x, *args], kwargs)


def cosine_similarity(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Cosine_similarity operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Cosine_similarity output.
    """
    return record_op("Cosine_similarity", [x, *args], kwargs)


def count_nonzero(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Count_nonzero operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Count_nonzero output.
    """
    return record_op("Count_nonzero", [x, *args], kwargs)


def cov(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Cov operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Cov output.
    """
    return record_op("Cov", [x, *args], kwargs)


def cross(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Cross operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Cross output.
    """
    return record_op("Cross", [x, *args], kwargs)


def crow_indices_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Crow_indices_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Crow_indices_copy output.
    """
    return record_op("Crow_indices_copy", [x, *args], kwargs)


def ctc_loss(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Ctc_loss operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Ctc_loss output.
    """
    return record_op("Ctc_loss", [x, *args], kwargs)


def cudnn_affine_grid_generator(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Cudnn_affine_grid_generator operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Cudnn_affine_grid_generator output.
    """
    return record_op("Cudnn_affine_grid_generator", [x, *args], kwargs)


def cudnn_batch_norm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Cudnn_batch_norm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Cudnn_batch_norm output.
    """
    return record_op("Cudnn_batch_norm", [x, *args], kwargs)


def cudnn_convolution(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Cudnn_convolution operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Cudnn_convolution output.
    """
    return record_op("Cudnn_convolution", [x, *args], kwargs)


def cudnn_convolution_add_relu(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Cudnn_convolution_add_relu operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Cudnn_convolution_add_relu output.
    """
    return record_op("Cudnn_convolution_add_relu", [x, *args], kwargs)


def cudnn_convolution_relu(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Cudnn_convolution_relu operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Cudnn_convolution_relu output.
    """
    return record_op("Cudnn_convolution_relu", [x, *args], kwargs)


def cudnn_convolution_transpose(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Cudnn_convolution_transpose operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Cudnn_convolution_transpose output.
    """
    return record_op("Cudnn_convolution_transpose", [x, *args], kwargs)


def cudnn_grid_sampler(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Cudnn_grid_sampler operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Cudnn_grid_sampler output.
    """
    return record_op("Cudnn_grid_sampler", [x, *args], kwargs)


def cummax(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Cummax operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Cummax output.
    """
    return record_op("Cummax", [x, *args], kwargs)


def cummin(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Cummin operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Cummin output.
    """
    return record_op("Cummin", [x, *args], kwargs)


def cumprod(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Cumprod operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Cumprod output.
    """
    return record_op("Cumprod", [x, *args], kwargs)


def cumulative_trapezoid(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Cumulative_trapezoid operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Cumulative_trapezoid output.
    """
    return record_op("Cumulative_trapezoid", [x, *args], kwargs)


def deg2rad(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Deg2rad operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Deg2rad output.
    """
    return record_op("Deg2rad", [x, *args], kwargs)


def deg2rad_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Deg2rad_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Deg2rad_ output.
    """
    return record_op("Deg2rad_", [x, *args], kwargs)


def dequantize(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Dequantize operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Dequantize output.
    """
    return record_op("Dequantize", [x, *args], kwargs)


def detach(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Detach operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Detach output.
    """
    return record_op("Detach", [x, *args], kwargs)


def detach_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Detach_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Detach_ output.
    """
    return record_op("Detach_", [x, *args], kwargs)


def detach_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Detach_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Detach_copy output.
    """
    return record_op("Detach_copy", [x, *args], kwargs)


def diag(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Diag operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Diag output.
    """
    return record_op("Diag", [x, *args], kwargs)


def diag_embed(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Diag_embed operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Diag_embed output.
    """
    return record_op("Diag_embed", [x, *args], kwargs)


def diagflat(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Diagflat operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Diagflat output.
    """
    return record_op("Diagflat", [x, *args], kwargs)


def diagonal(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Diagonal operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Diagonal output.
    """
    return record_op("Diagonal", [x, *args], kwargs)


def diagonal_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Diagonal_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Diagonal_copy output.
    """
    return record_op("Diagonal_copy", [x, *args], kwargs)


def diagonal_scatter(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Diagonal_scatter operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Diagonal_scatter output.
    """
    return record_op("Diagonal_scatter", [x, *args], kwargs)


def diff(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Diff operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Diff output.
    """
    return record_op("Diff", [x, *args], kwargs)


def digamma(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Digamma operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Digamma output.
    """
    return record_op("Digamma", [x, *args], kwargs)


def dist(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Dist operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Dist output.
    """
    return record_op("Dist", [x, *args], kwargs)


def divide(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Divide operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Divide output.
    """
    return record_op("Divide", [x, *args], kwargs)


def dot(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Dot operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Dot output.
    """
    return record_op("Dot", [x, *args], kwargs)


def dropout_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Dropout_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Dropout_ output.
    """
    return record_op("Dropout_", [x, *args], kwargs)


def dsmm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Dsmm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Dsmm output.
    """
    return record_op("Dsmm", [x, *args], kwargs)


def dsplit(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Dsplit operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Dsplit output.
    """
    return record_op("Dsplit", [x, *args], kwargs)


def dstack(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Dstack operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Dstack output.
    """
    return record_op("Dstack", [x, *args], kwargs)


def embedding(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Embedding operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Embedding output.
    """
    return record_op("Embedding", [x, *args], kwargs)


def embedding_bag(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Embedding_bag operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Embedding_bag output.
    """
    return record_op("Embedding_bag", [x, *args], kwargs)


def embedding_renorm_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Embedding_renorm_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Embedding_renorm_ output.
    """
    return record_op("Embedding_renorm_", [x, *args], kwargs)


def empty(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Empty operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Empty output.
    """
    return record_op("Empty", [x, *args], kwargs)


def empty_like(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Empty_like operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Empty_like output.
    """
    return record_op("Empty_like", [x, *args], kwargs)


def empty_permuted(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Empty_permuted operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Empty_permuted output.
    """
    return record_op("Empty_permuted", [x, *args], kwargs)


def empty_quantized(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Empty_quantized operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Empty_quantized output.
    """
    return record_op("Empty_quantized", [x, *args], kwargs)


def empty_strided(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Empty_strided operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Empty_strided output.
    """
    return record_op("Empty_strided", [x, *args], kwargs)


def eq(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Eq operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Eq output.
    """
    return record_op("Eq", [x, *args], kwargs)


def erf_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Erf_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Erf_ output.
    """
    return record_op("Erf_", [x, *args], kwargs)


def erfc(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Erfc operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Erfc output.
    """
    return record_op("Erfc", [x, *args], kwargs)


def erfc_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Erfc_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Erfc_ output.
    """
    return record_op("Erfc_", [x, *args], kwargs)


def erfinv(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Erfinv operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Erfinv output.
    """
    return record_op("Erfinv", [x, *args], kwargs)


def exp2(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Exp2 operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Exp2 output.
    """
    return record_op("Exp2", [x, *args], kwargs)


def exp2_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Exp2_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Exp2_ output.
    """
    return record_op("Exp2_", [x, *args], kwargs)


def exp_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Exp_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Exp_ output.
    """
    return record_op("Exp_", [x, *args], kwargs)


def expand_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Expand_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Expand_copy output.
    """
    return record_op("Expand_copy", [x, *args], kwargs)


def expm1(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Expm1 operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Expm1 output.
    """
    return record_op("Expm1", [x, *args], kwargs)


def expm1_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Expm1_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Expm1_ output.
    """
    return record_op("Expm1_", [x, *args], kwargs)


def eye(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Eye operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Eye output.
    """
    return record_op("Eye", [x, *args], kwargs)


def fake_quantize_per_channel_affine(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Fake_quantize_per_channel_affine operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Fake_quantize_per_channel_affine output.
    """
    return record_op("Fake_quantize_per_channel_affine", [x, *args], kwargs)


def fake_quantize_per_tensor_affine(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Fake_quantize_per_tensor_affine operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Fake_quantize_per_tensor_affine output.
    """
    return record_op("Fake_quantize_per_tensor_affine", [x, *args], kwargs)


def fbgemm_linear_fp16_weight(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Fbgemm_linear_fp16_weight operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Fbgemm_linear_fp16_weight output.
    """
    return record_op("Fbgemm_linear_fp16_weight", [x, *args], kwargs)


def fbgemm_linear_fp16_weight_fp32_activation(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Fbgemm_linear_fp16_weight_fp32_activation operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Fbgemm_linear_fp16_weight_fp32_activation output.
    """
    return record_op("Fbgemm_linear_fp16_weight_fp32_activation", [x, *args], kwargs)


def fbgemm_linear_int8_weight(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Fbgemm_linear_int8_weight operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Fbgemm_linear_int8_weight output.
    """
    return record_op("Fbgemm_linear_int8_weight", [x, *args], kwargs)


def fbgemm_linear_int8_weight_fp32_activation(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Fbgemm_linear_int8_weight_fp32_activation operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Fbgemm_linear_int8_weight_fp32_activation output.
    """
    return record_op("Fbgemm_linear_int8_weight_fp32_activation", [x, *args], kwargs)


def fbgemm_linear_quantize_weight(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Fbgemm_linear_quantize_weight operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Fbgemm_linear_quantize_weight output.
    """
    return record_op("Fbgemm_linear_quantize_weight", [x, *args], kwargs)


def fbgemm_pack_gemm_matrix_fp16(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Fbgemm_pack_gemm_matrix_fp16 operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Fbgemm_pack_gemm_matrix_fp16 output.
    """
    return record_op("Fbgemm_pack_gemm_matrix_fp16", [x, *args], kwargs)


def fbgemm_pack_quantized_matrix(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Fbgemm_pack_quantized_matrix operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Fbgemm_pack_quantized_matrix output.
    """
    return record_op("Fbgemm_pack_quantized_matrix", [x, *args], kwargs)


def feature_alpha_dropout(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Feature_alpha_dropout operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Feature_alpha_dropout output.
    """
    return record_op("Feature_alpha_dropout", [x, *args], kwargs)


def feature_alpha_dropout_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Feature_alpha_dropout_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Feature_alpha_dropout_ output.
    """
    return record_op("Feature_alpha_dropout_", [x, *args], kwargs)


def feature_dropout(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Feature_dropout operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Feature_dropout output.
    """
    return record_op("Feature_dropout", [x, *args], kwargs)


def feature_dropout_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Feature_dropout_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Feature_dropout_ output.
    """
    return record_op("Feature_dropout_", [x, *args], kwargs)


def fill(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Fill operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Fill output.
    """
    return record_op("Fill", [x, *args], kwargs)


def fill_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Fill_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Fill_ output.
    """
    return record_op("Fill_", [x, *args], kwargs)


def fix(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Fix operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Fix output.
    """
    return record_op("Fix", [x, *args], kwargs)


def fix_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Fix_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Fix_ output.
    """
    return record_op("Fix_", [x, *args], kwargs)


def flatten(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Flatten operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Flatten output.
    """
    return record_op("Flatten", [x, *args], kwargs)


def flip(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Flip operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Flip output.
    """
    return record_op("Flip", [x, *args], kwargs)


def fliplr(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Fliplr operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Fliplr output.
    """
    return record_op("Fliplr", [x, *args], kwargs)


def flipud(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Flipud operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Flipud output.
    """
    return record_op("Flipud", [x, *args], kwargs)


def float_power(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Float_power operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Float_power output.
    """
    return record_op("Float_power", [x, *args], kwargs)


def floor_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Floor_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Floor_ output.
    """
    return record_op("Floor_", [x, *args], kwargs)


def floor_divide(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Floor_divide operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Floor_divide output.
    """
    return record_op("Floor_divide", [x, *args], kwargs)


def fmax(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Fmax operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Fmax output.
    """
    return record_op("Fmax", [x, *args], kwargs)


def fmin(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Fmin operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Fmin output.
    """
    return record_op("Fmin", [x, *args], kwargs)


def fmod(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Fmod operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Fmod output.
    """
    return record_op("Fmod", [x, *args], kwargs)


def frac(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Frac operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Frac output.
    """
    return record_op("Frac", [x, *args], kwargs)


def frac_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Frac_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Frac_ output.
    """
    return record_op("Frac_", [x, *args], kwargs)


def frexp(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Frexp operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Frexp output.
    """
    return record_op("Frexp", [x, *args], kwargs)


def frobenius_norm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Frobenius_norm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Frobenius_norm output.
    """
    return record_op("Frobenius_norm", [x, *args], kwargs)


def from_file(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the From_file operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the From_file output.
    """
    return record_op("From_file", [x, *args], kwargs)


def from_numpy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the From_numpy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the From_numpy output.
    """
    return record_op("From_numpy", [x, *args], kwargs)


def frombuffer(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Frombuffer operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Frombuffer output.
    """
    return record_op("Frombuffer", [x, *args], kwargs)


def full(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Full operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Full output.
    """
    return record_op("Full", [x, *args], kwargs)


def full_like(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Full_like operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Full_like output.
    """
    return record_op("Full_like", [x, *args], kwargs)


def fused_moving_avg_obs_fake_quant(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Fused_moving_avg_obs_fake_quant operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Fused_moving_avg_obs_fake_quant output.
    """
    return record_op("Fused_moving_avg_obs_fake_quant", [x, *args], kwargs)


def gcd(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Gcd operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Gcd output.
    """
    return record_op("Gcd", [x, *args], kwargs)


def gcd_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Gcd_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Gcd_ output.
    """
    return record_op("Gcd_", [x, *args], kwargs)


def ge(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Ge operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Ge output.
    """
    return record_op("Ge", [x, *args], kwargs)


def geqrf(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Geqrf operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Geqrf output.
    """
    return record_op("Geqrf", [x, *args], kwargs)


def ger(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Ger operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Ger output.
    """
    return record_op("Ger", [x, *args], kwargs)


def gradient(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Gradient operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Gradient output.
    """
    return record_op("Gradient", [x, *args], kwargs)


def greater_equal(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Greater_equal operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Greater_equal output.
    """
    return record_op("Greater_equal", [x, *args], kwargs)


def grid_sampler(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Grid_sampler operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Grid_sampler output.
    """
    return record_op("Grid_sampler", [x, *args], kwargs)


def grid_sampler_2d(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Grid_sampler_2d operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Grid_sampler_2d output.
    """
    return record_op("Grid_sampler_2d", [x, *args], kwargs)


def grid_sampler_3d(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Grid_sampler_3d operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Grid_sampler_3d output.
    """
    return record_op("Grid_sampler_3d", [x, *args], kwargs)


def group_norm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Group_norm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Group_norm output.
    """
    return record_op("Group_norm", [x, *args], kwargs)


def gru_cell(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Gru_cell operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Gru_cell output.
    """
    return record_op("Gru_cell", [x, *args], kwargs)


def gt(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Gt operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Gt output.
    """
    return record_op("Gt", [x, *args], kwargs)


def hamming_window(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Hamming_window operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Hamming_window output.
    """
    return record_op("Hamming_window", [x, *args], kwargs)


def hann_window(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Hann_window operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Hann_window output.
    """
    return record_op("Hann_window", [x, *args], kwargs)


def hardshrink(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Hardshrink operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Hardshrink output.
    """
    return record_op("Hardshrink", [x, *args], kwargs)


def hash_tensor(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Hash_tensor operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Hash_tensor output.
    """
    return record_op("Hash_tensor", [x, *args], kwargs)


def heaviside(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Heaviside operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Heaviside output.
    """
    return record_op("Heaviside", [x, *args], kwargs)


def hinge_embedding_loss(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Hinge_embedding_loss operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Hinge_embedding_loss output.
    """
    return record_op("Hinge_embedding_loss", [x, *args], kwargs)


def histc(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Histc operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Histc output.
    """
    return record_op("Histc", [x, *args], kwargs)


def histogram(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Histogram operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Histogram output.
    """
    return record_op("Histogram", [x, *args], kwargs)


def histogramdd(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Histogramdd operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Histogramdd output.
    """
    return record_op("Histogramdd", [x, *args], kwargs)


def hsmm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Hsmm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Hsmm output.
    """
    return record_op("Hsmm", [x, *args], kwargs)


def hsplit(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Hsplit operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Hsplit output.
    """
    return record_op("Hsplit", [x, *args], kwargs)


def hspmm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Hspmm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Hspmm output.
    """
    return record_op("Hspmm", [x, *args], kwargs)


def hstack(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Hstack operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Hstack output.
    """
    return record_op("Hstack", [x, *args], kwargs)


def hypot(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Hypot operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Hypot output.
    """
    return record_op("Hypot", [x, *args], kwargs)


def i0(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the I0 operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the I0 output.
    """
    return record_op("I0", [x, *args], kwargs)


def i0_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the I0_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the I0_ output.
    """
    return record_op("I0_", [x, *args], kwargs)


def igamma(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Igamma operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Igamma output.
    """
    return record_op("Igamma", [x, *args], kwargs)


def igammac(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Igammac operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Igammac output.
    """
    return record_op("Igammac", [x, *args], kwargs)


def imag(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Imag operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Imag output.
    """
    return record_op("Imag", [x, *args], kwargs)


def index_add(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Index_add operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Index_add output.
    """
    return record_op("Index_add", [x, *args], kwargs)


def index_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Index_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Index_copy output.
    """
    return record_op("Index_copy", [x, *args], kwargs)


def index_fill(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Index_fill operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Index_fill output.
    """
    return record_op("Index_fill", [x, *args], kwargs)


def index_put(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Index_put operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Index_put output.
    """
    return record_op("Index_put", [x, *args], kwargs)


def index_put_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Index_put_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Index_put_ output.
    """
    return record_op("Index_put_", [x, *args], kwargs)


def index_reduce(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Index_reduce operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Index_reduce output.
    """
    return record_op("Index_reduce", [x, *args], kwargs)


def index_select(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Index_select operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Index_select output.
    """
    return record_op("Index_select", [x, *args], kwargs)


def indices_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Indices_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Indices_copy output.
    """
    return record_op("Indices_copy", [x, *args], kwargs)


def inner(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Inner operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Inner output.
    """
    return record_op("Inner", [x, *args], kwargs)


def instance_norm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Instance_norm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Instance_norm output.
    """
    return record_op("Instance_norm", [x, *args], kwargs)


def int_repr(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Int_repr operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Int_repr output.
    """
    return record_op("Int_repr", [x, *args], kwargs)


def inverse(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Inverse operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Inverse output.
    """
    return record_op("Inverse", [x, *args], kwargs)


def is_complex(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Is_complex operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Is_complex output.
    """
    return record_op("Is_complex", [x, *args], kwargs)


def is_conj(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Is_conj operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Is_conj output.
    """
    return record_op("Is_conj", [x, *args], kwargs)


def is_floating_point(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Is_floating_point operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Is_floating_point output.
    """
    return record_op("Is_floating_point", [x, *args], kwargs)


def is_neg(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Is_neg operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Is_neg output.
    """
    return record_op("Is_neg", [x, *args], kwargs)


def is_nonzero(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Is_nonzero operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Is_nonzero output.
    """
    return record_op("Is_nonzero", [x, *args], kwargs)


def is_same_size(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Is_same_size operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Is_same_size output.
    """
    return record_op("Is_same_size", [x, *args], kwargs)


def is_signed(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Is_signed operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Is_signed output.
    """
    return record_op("Is_signed", [x, *args], kwargs)


def isclose(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Isclose operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Isclose output.
    """
    return record_op("Isclose", [x, *args], kwargs)


def isfinite(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Isfinite operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Isfinite output.
    """
    return record_op("Isfinite", [x, *args], kwargs)


def isin(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Isin operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Isin output.
    """
    return record_op("Isin", [x, *args], kwargs)


def isneginf(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Isneginf operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Isneginf output.
    """
    return record_op("Isneginf", [x, *args], kwargs)


def isposinf(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Isposinf operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Isposinf output.
    """
    return record_op("Isposinf", [x, *args], kwargs)


def isreal(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Isreal operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Isreal output.
    """
    return record_op("Isreal", [x, *args], kwargs)


def istft(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Istft operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Istft output.
    """
    return record_op("Istft", [x, *args], kwargs)


def kaiser_window(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Kaiser_window operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Kaiser_window output.
    """
    return record_op("Kaiser_window", [x, *args], kwargs)


def kl_div(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Kl_div operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Kl_div output.
    """
    return record_op("Kl_div", [x, *args], kwargs)


def kron(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Kron operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Kron output.
    """
    return record_op("Kron", [x, *args], kwargs)


def kthvalue(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Kthvalue operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Kthvalue output.
    """
    return record_op("Kthvalue", [x, *args], kwargs)


def layer_norm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Layer_norm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Layer_norm output.
    """
    return record_op("Layer_norm", [x, *args], kwargs)


def lcm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Lcm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Lcm output.
    """
    return record_op("Lcm", [x, *args], kwargs)


def lcm_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Lcm_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Lcm_ output.
    """
    return record_op("Lcm_", [x, *args], kwargs)


def ldexp(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Ldexp operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Ldexp output.
    """
    return record_op("Ldexp", [x, *args], kwargs)


def ldexp_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Ldexp_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Ldexp_ output.
    """
    return record_op("Ldexp_", [x, *args], kwargs)


def le(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Le operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Le output.
    """
    return record_op("Le", [x, *args], kwargs)


def lerp(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Lerp operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Lerp output.
    """
    return record_op("Lerp", [x, *args], kwargs)


def less_equal(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Less_equal operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Less_equal output.
    """
    return record_op("Less_equal", [x, *args], kwargs)


def lgamma(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Lgamma operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Lgamma output.
    """
    return record_op("Lgamma", [x, *args], kwargs)


def linspace(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Linspace operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Linspace output.
    """
    return record_op("Linspace", [x, *args], kwargs)


def log(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Log operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Log output.
    """
    return record_op("Log", [x, *args], kwargs)


def log10(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Log10 operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Log10 output.
    """
    return record_op("Log10", [x, *args], kwargs)


def log10_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Log10_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Log10_ output.
    """
    return record_op("Log10_", [x, *args], kwargs)


def log1p(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Log1p operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Log1p output.
    """
    return record_op("Log1p", [x, *args], kwargs)


def log1p_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Log1p_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Log1p_ output.
    """
    return record_op("Log1p_", [x, *args], kwargs)


def log2(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Log2 operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Log2 output.
    """
    return record_op("Log2", [x, *args], kwargs)


def log2_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Log2_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Log2_ output.
    """
    return record_op("Log2_", [x, *args], kwargs)


def log_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Log_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Log_ output.
    """
    return record_op("Log_", [x, *args], kwargs)


def logaddexp(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Logaddexp operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Logaddexp output.
    """
    return record_op("Logaddexp", [x, *args], kwargs)


def logaddexp2(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Logaddexp2 operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Logaddexp2 output.
    """
    return record_op("Logaddexp2", [x, *args], kwargs)


def logcumsumexp(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Logcumsumexp operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Logcumsumexp output.
    """
    return record_op("Logcumsumexp", [x, *args], kwargs)


def logdet(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Logdet operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Logdet output.
    """
    return record_op("Logdet", [x, *args], kwargs)


def logical_and(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Logical_and operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Logical_and output.
    """
    return record_op("Logical_and", [x, *args], kwargs)


def logical_not(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Logical_not operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Logical_not output.
    """
    return record_op("Logical_not", [x, *args], kwargs)


def logical_or(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Logical_or operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Logical_or output.
    """
    return record_op("Logical_or", [x, *args], kwargs)


def logical_xor(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Logical_xor operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Logical_xor output.
    """
    return record_op("Logical_xor", [x, *args], kwargs)


def logit(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Logit operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Logit output.
    """
    return record_op("Logit", [x, *args], kwargs)


def logit_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Logit_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Logit_ output.
    """
    return record_op("Logit_", [x, *args], kwargs)


def logspace(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Logspace operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Logspace output.
    """
    return record_op("Logspace", [x, *args], kwargs)


def logsumexp(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Logsumexp operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Logsumexp output.
    """
    return record_op("Logsumexp", [x, *args], kwargs)


def lstm_cell(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Lstm_cell operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Lstm_cell output.
    """
    return record_op("Lstm_cell", [x, *args], kwargs)


def lt(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Lt operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Lt output.
    """
    return record_op("Lt", [x, *args], kwargs)


def lu_solve(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Lu_solve operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Lu_solve output.
    """
    return record_op("Lu_solve", [x, *args], kwargs)


def lu_unpack(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Lu_unpack operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Lu_unpack output.
    """
    return record_op("Lu_unpack", [x, *args], kwargs)


def margin_ranking_loss(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Margin_ranking_loss operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Margin_ranking_loss output.
    """
    return record_op("Margin_ranking_loss", [x, *args], kwargs)


def masked_fill(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Masked_fill operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Masked_fill output.
    """
    return record_op("Masked_fill", [x, *args], kwargs)


def masked_scatter(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Masked_scatter operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Masked_scatter output.
    """
    return record_op("Masked_scatter", [x, *args], kwargs)


def masked_select(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Masked_select operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Masked_select output.
    """
    return record_op("Masked_select", [x, *args], kwargs)


def matrix_exp(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Matrix_exp operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Matrix_exp output.
    """
    return record_op("Matrix_exp", [x, *args], kwargs)


def matrix_power(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Matrix_power operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Matrix_power output.
    """
    return record_op("Matrix_power", [x, *args], kwargs)


def max_pool1d(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Max_pool1d operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Max_pool1d output.
    """
    return record_op("Max_pool1d", [x, *args], kwargs)


def max_pool1d_with_indices(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Max_pool1d_with_indices operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Max_pool1d_with_indices output.
    """
    return record_op("Max_pool1d_with_indices", [x, *args], kwargs)


def max_pool2d(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Max_pool2d operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Max_pool2d output.
    """
    return record_op("Max_pool2d", [x, *args], kwargs)


def max_pool3d(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Max_pool3d operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Max_pool3d output.
    """
    return record_op("Max_pool3d", [x, *args], kwargs)


def maximum(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Maximum operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Maximum output.
    """
    return record_op("Maximum", [x, *args], kwargs)


def median(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Median operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Median output.
    """
    return record_op("Median", [x, *args], kwargs)


def meshgrid(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Meshgrid operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Meshgrid output.
    """
    return record_op("Meshgrid", [x, *args], kwargs)


def minimum(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Minimum operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Minimum output.
    """
    return record_op("Minimum", [x, *args], kwargs)


def miopen_batch_norm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Miopen_batch_norm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Miopen_batch_norm output.
    """
    return record_op("Miopen_batch_norm", [x, *args], kwargs)


def miopen_convolution(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Miopen_convolution operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Miopen_convolution output.
    """
    return record_op("Miopen_convolution", [x, *args], kwargs)


def miopen_convolution_add_relu(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Miopen_convolution_add_relu operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Miopen_convolution_add_relu output.
    """
    return record_op("Miopen_convolution_add_relu", [x, *args], kwargs)


def miopen_convolution_relu(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Miopen_convolution_relu operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Miopen_convolution_relu output.
    """
    return record_op("Miopen_convolution_relu", [x, *args], kwargs)


def miopen_convolution_transpose(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Miopen_convolution_transpose operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Miopen_convolution_transpose output.
    """
    return record_op("Miopen_convolution_transpose", [x, *args], kwargs)


def miopen_ctc_loss(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Miopen_ctc_loss operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Miopen_ctc_loss output.
    """
    return record_op("Miopen_ctc_loss", [x, *args], kwargs)


def miopen_depthwise_convolution(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Miopen_depthwise_convolution operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Miopen_depthwise_convolution output.
    """
    return record_op("Miopen_depthwise_convolution", [x, *args], kwargs)


def miopen_rnn(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Miopen_rnn operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Miopen_rnn output.
    """
    return record_op("Miopen_rnn", [x, *args], kwargs)


def mkldnn_adaptive_avg_pool2d(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Mkldnn_adaptive_avg_pool2d operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Mkldnn_adaptive_avg_pool2d output.
    """
    return record_op("Mkldnn_adaptive_avg_pool2d", [x, *args], kwargs)


def mkldnn_convolution(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Mkldnn_convolution operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Mkldnn_convolution output.
    """
    return record_op("Mkldnn_convolution", [x, *args], kwargs)


def mkldnn_linear_backward_weights(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Mkldnn_linear_backward_weights operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Mkldnn_linear_backward_weights output.
    """
    return record_op("Mkldnn_linear_backward_weights", [x, *args], kwargs)


def mkldnn_max_pool2d(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Mkldnn_max_pool2d operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Mkldnn_max_pool2d output.
    """
    return record_op("Mkldnn_max_pool2d", [x, *args], kwargs)


def mkldnn_max_pool3d(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Mkldnn_max_pool3d operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Mkldnn_max_pool3d output.
    """
    return record_op("Mkldnn_max_pool3d", [x, *args], kwargs)


def mkldnn_rnn_layer(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Mkldnn_rnn_layer operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Mkldnn_rnn_layer output.
    """
    return record_op("Mkldnn_rnn_layer", [x, *args], kwargs)


def mm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Mm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Mm output.
    """
    return record_op("Mm", [x, *args], kwargs)


def mode(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Mode operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Mode output.
    """
    return record_op("Mode", [x, *args], kwargs)


def moveaxis(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Moveaxis operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Moveaxis output.
    """
    return record_op("Moveaxis", [x, *args], kwargs)


def movedim(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Movedim operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Movedim output.
    """
    return record_op("Movedim", [x, *args], kwargs)


def msort(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Msort operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Msort output.
    """
    return record_op("Msort", [x, *args], kwargs)


def multiply(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Multiply operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Multiply output.
    """
    return record_op("Multiply", [x, *args], kwargs)


def mv(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Mv operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Mv output.
    """
    return record_op("Mv", [x, *args], kwargs)


def mvlgamma(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Mvlgamma operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Mvlgamma output.
    """
    return record_op("Mvlgamma", [x, *args], kwargs)


def nan_to_num(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Nan_to_num operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Nan_to_num output.
    """
    return record_op("Nan_to_num", [x, *args], kwargs)


def nan_to_num_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Nan_to_num_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Nan_to_num_ output.
    """
    return record_op("Nan_to_num_", [x, *args], kwargs)


def nanmean(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Nanmean operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Nanmean output.
    """
    return record_op("Nanmean", [x, *args], kwargs)


def nanmedian(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Nanmedian operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Nanmedian output.
    """
    return record_op("Nanmedian", [x, *args], kwargs)


def nanquantile(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Nanquantile operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Nanquantile output.
    """
    return record_op("Nanquantile", [x, *args], kwargs)


def nansum(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Nansum operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Nansum output.
    """
    return record_op("Nansum", [x, *args], kwargs)


def narrow(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Narrow operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Narrow output.
    """
    return record_op("Narrow", [x, *args], kwargs)


def narrow_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Narrow_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Narrow_copy output.
    """
    return record_op("Narrow_copy", [x, *args], kwargs)


def native_batch_norm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Native_batch_norm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Native_batch_norm output.
    """
    return record_op("Native_batch_norm", [x, *args], kwargs)


def native_channel_shuffle(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Native_channel_shuffle operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Native_channel_shuffle output.
    """
    return record_op("Native_channel_shuffle", [x, *args], kwargs)


def native_dropout(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Native_dropout operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Native_dropout output.
    """
    return record_op("Native_dropout", [x, *args], kwargs)


def native_group_norm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Native_group_norm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Native_group_norm output.
    """
    return record_op("Native_group_norm", [x, *args], kwargs)


def native_layer_norm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Native_layer_norm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Native_layer_norm output.
    """
    return record_op("Native_layer_norm", [x, *args], kwargs)


def native_norm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Native_norm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Native_norm output.
    """
    return record_op("Native_norm", [x, *args], kwargs)


def ne(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Ne operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Ne output.
    """
    return record_op("Ne", [x, *args], kwargs)


def neg_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Neg_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Neg_ output.
    """
    return record_op("Neg_", [x, *args], kwargs)


def negative(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Negative operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Negative output.
    """
    return record_op("Negative", [x, *args], kwargs)


def negative_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Negative_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Negative_ output.
    """
    return record_op("Negative_", [x, *args], kwargs)


def nextafter(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Nextafter operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Nextafter output.
    """
    return record_op("Nextafter", [x, *args], kwargs)


def nonzero(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Nonzero operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Nonzero output.
    """
    return record_op("Nonzero", [x, *args], kwargs)


def nonzero_static(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Nonzero_static operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Nonzero_static output.
    """
    return record_op("Nonzero_static", [x, *args], kwargs)


def norm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Norm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Norm output.
    """
    return record_op("Norm", [x, *args], kwargs)


def norm_except_dim(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Norm_except_dim operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Norm_except_dim output.
    """
    return record_op("Norm_except_dim", [x, *args], kwargs)


def normal(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Normal operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Normal output.
    """
    return record_op("Normal", [x, *args], kwargs)


def not_equal(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Not_equal operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Not_equal output.
    """
    return record_op("Not_equal", [x, *args], kwargs)


def nuclear_norm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Nuclear_norm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Nuclear_norm output.
    """
    return record_op("Nuclear_norm", [x, *args], kwargs)


def numel(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Numel operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Numel output.
    """
    return record_op("Numel", [x, *args], kwargs)


def ones_like(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Ones_like operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Ones_like output.
    """
    return record_op("Ones_like", [x, *args], kwargs)


def orgqr(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Orgqr operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Orgqr output.
    """
    return record_op("Orgqr", [x, *args], kwargs)


def ormqr(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Ormqr operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Ormqr output.
    """
    return record_op("Ormqr", [x, *args], kwargs)


def outer(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Outer operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Outer output.
    """
    return record_op("Outer", [x, *args], kwargs)


def pairwise_distance(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Pairwise_distance operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Pairwise_distance output.
    """
    return record_op("Pairwise_distance", [x, *args], kwargs)


def pdist(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Pdist operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Pdist output.
    """
    return record_op("Pdist", [x, *args], kwargs)


def permute(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Permute operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Permute output.
    """
    return record_op("Permute", [x, *args], kwargs)


def permute_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Permute_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Permute_copy output.
    """
    return record_op("Permute_copy", [x, *args], kwargs)


def pinverse(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Pinverse operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Pinverse output.
    """
    return record_op("Pinverse", [x, *args], kwargs)


def pixel_shuffle(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Pixel_shuffle operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Pixel_shuffle output.
    """
    return record_op("Pixel_shuffle", [x, *args], kwargs)


def pixel_unshuffle(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Pixel_unshuffle operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Pixel_unshuffle output.
    """
    return record_op("Pixel_unshuffle", [x, *args], kwargs)


def poisson(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Poisson operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Poisson output.
    """
    return record_op("Poisson", [x, *args], kwargs)


def poisson_nll_loss(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Poisson_nll_loss operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Poisson_nll_loss output.
    """
    return record_op("Poisson_nll_loss", [x, *args], kwargs)


def polar(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Polar operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Polar output.
    """
    return record_op("Polar", [x, *args], kwargs)


def polygamma(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Polygamma operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Polygamma output.
    """
    return record_op("Polygamma", [x, *args], kwargs)


def positive(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Positive operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Positive output.
    """
    return record_op("Positive", [x, *args], kwargs)


def prod(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Prod operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Prod output.
    """
    return record_op("Prod", [x, *args], kwargs)


def put(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Put operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Put output.
    """
    return record_op("Put", [x, *args], kwargs)


def q_per_channel_axis(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Q_per_channel_axis operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Q_per_channel_axis output.
    """
    return record_op("Q_per_channel_axis", [x, *args], kwargs)


def q_per_channel_scales(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Q_per_channel_scales operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Q_per_channel_scales output.
    """
    return record_op("Q_per_channel_scales", [x, *args], kwargs)


def q_per_channel_zero_points(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Q_per_channel_zero_points operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Q_per_channel_zero_points output.
    """
    return record_op("Q_per_channel_zero_points", [x, *args], kwargs)


def q_scale(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Q_scale operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Q_scale output.
    """
    return record_op("Q_scale", [x, *args], kwargs)


def q_zero_point(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Q_zero_point operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Q_zero_point output.
    """
    return record_op("Q_zero_point", [x, *args], kwargs)


def qr(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Qr operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Qr output.
    """
    return record_op("Qr", [x, *args], kwargs)


def quantile(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Quantile operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Quantile output.
    """
    return record_op("Quantile", [x, *args], kwargs)


def quantize_per_channel(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Quantize_per_channel operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Quantize_per_channel output.
    """
    return record_op("Quantize_per_channel", [x, *args], kwargs)


def quantize_per_tensor(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Quantize_per_tensor operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Quantize_per_tensor output.
    """
    return record_op("Quantize_per_tensor", [x, *args], kwargs)


def quantize_per_tensor_dynamic(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Quantize_per_tensor_dynamic operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Quantize_per_tensor_dynamic output.
    """
    return record_op("Quantize_per_tensor_dynamic", [x, *args], kwargs)


def quantized_batch_norm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Quantized_batch_norm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Quantized_batch_norm output.
    """
    return record_op("Quantized_batch_norm", [x, *args], kwargs)


def quantized_gru_cell(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Quantized_gru_cell operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Quantized_gru_cell output.
    """
    return record_op("Quantized_gru_cell", [x, *args], kwargs)


def quantized_lstm_cell(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Quantized_lstm_cell operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Quantized_lstm_cell output.
    """
    return record_op("Quantized_lstm_cell", [x, *args], kwargs)


def quantized_max_pool1d(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Quantized_max_pool1d operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Quantized_max_pool1d output.
    """
    return record_op("Quantized_max_pool1d", [x, *args], kwargs)


def quantized_max_pool2d(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Quantized_max_pool2d operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Quantized_max_pool2d output.
    """
    return record_op("Quantized_max_pool2d", [x, *args], kwargs)


def quantized_max_pool3d(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Quantized_max_pool3d operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Quantized_max_pool3d output.
    """
    return record_op("Quantized_max_pool3d", [x, *args], kwargs)


def quantized_rnn_relu_cell(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Quantized_rnn_relu_cell operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Quantized_rnn_relu_cell output.
    """
    return record_op("Quantized_rnn_relu_cell", [x, *args], kwargs)


def quantized_rnn_tanh_cell(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Quantized_rnn_tanh_cell operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Quantized_rnn_tanh_cell output.
    """
    return record_op("Quantized_rnn_tanh_cell", [x, *args], kwargs)


def rad2deg(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Rad2deg operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Rad2deg output.
    """
    return record_op("Rad2deg", [x, *args], kwargs)


def rad2deg_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Rad2deg_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Rad2deg_ output.
    """
    return record_op("Rad2deg_", [x, *args], kwargs)


def rand_like(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Rand_like operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Rand_like output.
    """
    return record_op("Rand_like", [x, *args], kwargs)


def randint(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Randint operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Randint output.
    """
    return record_op("Randint", [x, *args], kwargs)


def randint_like(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Randint_like operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Randint_like output.
    """
    return record_op("Randint_like", [x, *args], kwargs)


def randn_like(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Randn_like operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Randn_like output.
    """
    return record_op("Randn_like", [x, *args], kwargs)


def randperm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Randperm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Randperm output.
    """
    return record_op("Randperm", [x, *args], kwargs)


def range(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Range operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Range output.
    """
    return record_op("Range", [x, *args], kwargs)


def ravel(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Ravel operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Ravel output.
    """
    return record_op("Ravel", [x, *args], kwargs)


def real(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Real operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Real output.
    """
    return record_op("Real", [x, *args], kwargs)


def reciprocal_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Reciprocal_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Reciprocal_ output.
    """
    return record_op("Reciprocal_", [x, *args], kwargs)


def relu_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Relu_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Relu_ output.
    """
    return record_op("Relu_", [x, *args], kwargs)


def remainder(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Remainder operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Remainder output.
    """
    return record_op("Remainder", [x, *args], kwargs)


def renorm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Renorm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Renorm output.
    """
    return record_op("Renorm", [x, *args], kwargs)


def repeat_interleave(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Repeat_interleave operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Repeat_interleave output.
    """
    return record_op("Repeat_interleave", [x, *args], kwargs)


def resize_as_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Resize_as_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Resize_as_ output.
    """
    return record_op("Resize_as_", [x, *args], kwargs)


def resize_as_sparse_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Resize_as_sparse_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Resize_as_sparse_ output.
    """
    return record_op("Resize_as_sparse_", [x, *args], kwargs)


def resolve_conj(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Resolve_conj operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Resolve_conj output.
    """
    return record_op("Resolve_conj", [x, *args], kwargs)


def resolve_neg(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Resolve_neg operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Resolve_neg output.
    """
    return record_op("Resolve_neg", [x, *args], kwargs)


def rms_norm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Rms_norm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Rms_norm output.
    """
    return record_op("Rms_norm", [x, *args], kwargs)


def rnn_relu(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Rnn_relu operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Rnn_relu output.
    """
    return record_op("Rnn_relu", [x, *args], kwargs)


def rnn_relu_cell(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Rnn_relu_cell operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Rnn_relu_cell output.
    """
    return record_op("Rnn_relu_cell", [x, *args], kwargs)


def rnn_tanh(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Rnn_tanh operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Rnn_tanh output.
    """
    return record_op("Rnn_tanh", [x, *args], kwargs)


def rnn_tanh_cell(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Rnn_tanh_cell operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Rnn_tanh_cell output.
    """
    return record_op("Rnn_tanh_cell", [x, *args], kwargs)


def roll(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Roll operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Roll output.
    """
    return record_op("Roll", [x, *args], kwargs)


def rot90(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Rot90 operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Rot90 output.
    """
    return record_op("Rot90", [x, *args], kwargs)


def round_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Round_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Round_ output.
    """
    return record_op("Round_", [x, *args], kwargs)


def row_indices_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Row_indices_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Row_indices_copy output.
    """
    return record_op("Row_indices_copy", [x, *args], kwargs)


def row_stack(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Row_stack operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Row_stack output.
    """
    return record_op("Row_stack", [x, *args], kwargs)


def rrelu(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Rrelu operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Rrelu output.
    """
    return record_op("Rrelu", [x, *args], kwargs)


def rrelu_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Rrelu_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Rrelu_ output.
    """
    return record_op("Rrelu_", [x, *args], kwargs)


def rsqrt(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Rsqrt operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Rsqrt output.
    """
    return record_op("Rsqrt", [x, *args], kwargs)


def rsqrt_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Rsqrt_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Rsqrt_ output.
    """
    return record_op("Rsqrt_", [x, *args], kwargs)


def rsub(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Rsub operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Rsub output.
    """
    return record_op("Rsub", [x, *args], kwargs)


def saddmm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Saddmm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Saddmm output.
    """
    return record_op("Saddmm", [x, *args], kwargs)


def scalar_tensor(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Scalar_tensor operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Scalar_tensor output.
    """
    return record_op("Scalar_tensor", [x, *args], kwargs)


def scatter_add(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Scatter_add operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Scatter_add output.
    """
    return record_op("Scatter_add", [x, *args], kwargs)


def scatter_reduce(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Scatter_reduce operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Scatter_reduce output.
    """
    return record_op("Scatter_reduce", [x, *args], kwargs)


def searchsorted(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Searchsorted operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Searchsorted output.
    """
    return record_op("Searchsorted", [x, *args], kwargs)


def select(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Select operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Select output.
    """
    return record_op("Select", [x, *args], kwargs)


def select_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Select_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Select_copy output.
    """
    return record_op("Select_copy", [x, *args], kwargs)


def select_scatter(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Select_scatter operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Select_scatter output.
    """
    return record_op("Select_scatter", [x, *args], kwargs)


def selu_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Selu_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Selu_ output.
    """
    return record_op("Selu_", [x, *args], kwargs)


def sgn(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Sgn operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Sgn output.
    """
    return record_op("Sgn", [x, *args], kwargs)


def sigmoid_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Sigmoid_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Sigmoid_ output.
    """
    return record_op("Sigmoid_", [x, *args], kwargs)


def signbit(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Signbit operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Signbit output.
    """
    return record_op("Signbit", [x, *args], kwargs)


def sin_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Sin_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Sin_ output.
    """
    return record_op("Sin_", [x, *args], kwargs)


def sinc(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Sinc operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Sinc output.
    """
    return record_op("Sinc", [x, *args], kwargs)


def sinc_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Sinc_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Sinc_ output.
    """
    return record_op("Sinc_", [x, *args], kwargs)


def sinh_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Sinh_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Sinh_ output.
    """
    return record_op("Sinh_", [x, *args], kwargs)


def slice_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Slice_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Slice_copy output.
    """
    return record_op("Slice_copy", [x, *args], kwargs)


def slice_inverse(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Slice_inverse operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Slice_inverse output.
    """
    return record_op("Slice_inverse", [x, *args], kwargs)


def slice_scatter(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Slice_scatter operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Slice_scatter output.
    """
    return record_op("Slice_scatter", [x, *args], kwargs)


def slogdet(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Slogdet operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Slogdet output.
    """
    return record_op("Slogdet", [x, *args], kwargs)


def smm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Smm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Smm output.
    """
    return record_op("Smm", [x, *args], kwargs)


def sort(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Sort operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Sort output.
    """
    return record_op("Sort", [x, *args], kwargs)


def sparse_bsc_tensor(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Sparse_bsc_tensor operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Sparse_bsc_tensor output.
    """
    return record_op("Sparse_bsc_tensor", [x, *args], kwargs)


def sparse_bsr_tensor(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Sparse_bsr_tensor operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Sparse_bsr_tensor output.
    """
    return record_op("Sparse_bsr_tensor", [x, *args], kwargs)


def sparse_compressed_tensor(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Sparse_compressed_tensor operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Sparse_compressed_tensor output.
    """
    return record_op("Sparse_compressed_tensor", [x, *args], kwargs)


def sparse_coo_tensor(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Sparse_coo_tensor operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Sparse_coo_tensor output.
    """
    return record_op("Sparse_coo_tensor", [x, *args], kwargs)


def sparse_csc_tensor(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Sparse_csc_tensor operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Sparse_csc_tensor output.
    """
    return record_op("Sparse_csc_tensor", [x, *args], kwargs)


def sparse_csr_tensor(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Sparse_csr_tensor operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Sparse_csr_tensor output.
    """
    return record_op("Sparse_csr_tensor", [x, *args], kwargs)


def split_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Split_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Split_copy output.
    """
    return record_op("Split_copy", [x, *args], kwargs)


def split_with_sizes(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Split_with_sizes operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Split_with_sizes output.
    """
    return record_op("Split_with_sizes", [x, *args], kwargs)


def split_with_sizes_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Split_with_sizes_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Split_with_sizes_copy output.
    """
    return record_op("Split_with_sizes_copy", [x, *args], kwargs)


def spmm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Spmm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Spmm output.
    """
    return record_op("Spmm", [x, *args], kwargs)


def sqrt(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Sqrt operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Sqrt output.
    """
    return record_op("Sqrt", [x, *args], kwargs)


def sqrt_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Sqrt_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Sqrt_ output.
    """
    return record_op("Sqrt_", [x, *args], kwargs)


def square(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Square operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Square output.
    """
    return record_op("Square", [x, *args], kwargs)


def square_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Square_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Square_ output.
    """
    return record_op("Square_", [x, *args], kwargs)


def squeeze(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Squeeze operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Squeeze output.
    """
    return record_op("Squeeze", [x, *args], kwargs)


def squeeze_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Squeeze_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Squeeze_copy output.
    """
    return record_op("Squeeze_copy", [x, *args], kwargs)


def sspaddmm(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Sspaddmm operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Sspaddmm output.
    """
    return record_op("Sspaddmm", [x, *args], kwargs)


def std(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Std operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Std output.
    """
    return record_op("Std", [x, *args], kwargs)


def std_mean(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Std_mean operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Std_mean output.
    """
    return record_op("Std_mean", [x, *args], kwargs)


def stft(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Stft operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Stft output.
    """
    return record_op("Stft", [x, *args], kwargs)


def subtract(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Subtract operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Subtract output.
    """
    return record_op("Subtract", [x, *args], kwargs)


def svd(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Svd operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Svd output.
    """
    return record_op("Svd", [x, *args], kwargs)


def swapaxes(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Swapaxes operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Swapaxes output.
    """
    return record_op("Swapaxes", [x, *args], kwargs)


def swapdims(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Swapdims operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Swapdims output.
    """
    return record_op("Swapdims", [x, *args], kwargs)


def t(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the T operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the T output.
    """
    return record_op("T", [x, *args], kwargs)


def t_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the T_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the T_copy output.
    """
    return record_op("T_copy", [x, *args], kwargs)


def take(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Take operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Take output.
    """
    return record_op("Take", [x, *args], kwargs)


def take_along_dim(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Take_along_dim operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Take_along_dim output.
    """
    return record_op("Take_along_dim", [x, *args], kwargs)


def tan_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Tan_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Tan_ output.
    """
    return record_op("Tan_", [x, *args], kwargs)


def tanh_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Tanh_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Tanh_ output.
    """
    return record_op("Tanh_", [x, *args], kwargs)


def tensor_split(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Tensor_split operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Tensor_split output.
    """
    return record_op("Tensor_split", [x, *args], kwargs)


def tensordot(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Tensordot operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Tensordot output.
    """
    return record_op("Tensordot", [x, *args], kwargs)


def threshold(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Threshold operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Threshold output.
    """
    return record_op("Threshold", [x, *args], kwargs)


def threshold_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Threshold_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Threshold_ output.
    """
    return record_op("Threshold_", [x, *args], kwargs)


def transpose_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Transpose_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Transpose_copy output.
    """
    return record_op("Transpose_copy", [x, *args], kwargs)


def trapezoid(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Trapezoid operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Trapezoid output.
    """
    return record_op("Trapezoid", [x, *args], kwargs)


def trapz(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Trapz operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Trapz output.
    """
    return record_op("Trapz", [x, *args], kwargs)


def triangular_solve(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Triangular_solve operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Triangular_solve output.
    """
    return record_op("Triangular_solve", [x, *args], kwargs)


def tril(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Tril operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Tril output.
    """
    return record_op("Tril", [x, *args], kwargs)


def tril_indices(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Tril_indices operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Tril_indices output.
    """
    return record_op("Tril_indices", [x, *args], kwargs)


def triplet_margin_loss(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Triplet_margin_loss operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Triplet_margin_loss output.
    """
    return record_op("Triplet_margin_loss", [x, *args], kwargs)


def triu(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Triu operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Triu output.
    """
    return record_op("Triu", [x, *args], kwargs)


def triu_indices(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Triu_indices operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Triu_indices output.
    """
    return record_op("Triu_indices", [x, *args], kwargs)


def true_divide(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the True_divide operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the True_divide output.
    """
    return record_op("True_divide", [x, *args], kwargs)


def trunc(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Trunc operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Trunc output.
    """
    return record_op("Trunc", [x, *args], kwargs)


def trunc_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Trunc_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Trunc_ output.
    """
    return record_op("Trunc_", [x, *args], kwargs)


def unbind(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Unbind operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Unbind output.
    """
    return record_op("Unbind", [x, *args], kwargs)


def unbind_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Unbind_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Unbind_copy output.
    """
    return record_op("Unbind_copy", [x, *args], kwargs)


def unflatten(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Unflatten operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Unflatten output.
    """
    return record_op("Unflatten", [x, *args], kwargs)


def unfold_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Unfold_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Unfold_copy output.
    """
    return record_op("Unfold_copy", [x, *args], kwargs)


def unique_consecutive(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Unique_consecutive operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Unique_consecutive output.
    """
    return record_op("Unique_consecutive", [x, *args], kwargs)


def unsafe_chunk(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Unsafe_chunk operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Unsafe_chunk output.
    """
    return record_op("Unsafe_chunk", [x, *args], kwargs)


def unsafe_split(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Unsafe_split operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Unsafe_split output.
    """
    return record_op("Unsafe_split", [x, *args], kwargs)


def unsafe_split_with_sizes(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Unsafe_split_with_sizes operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Unsafe_split_with_sizes output.
    """
    return record_op("Unsafe_split_with_sizes", [x, *args], kwargs)


def unsqueeze(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Unsqueeze operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Unsqueeze output.
    """
    return record_op("Unsqueeze", [x, *args], kwargs)


def unsqueeze_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Unsqueeze_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Unsqueeze_copy output.
    """
    return record_op("Unsqueeze_copy", [x, *args], kwargs)


def values_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Values_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Values_copy output.
    """
    return record_op("Values_copy", [x, *args], kwargs)


def vander(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Vander operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Vander output.
    """
    return record_op("Vander", [x, *args], kwargs)


def var(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Var operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Var output.
    """
    return record_op("Var", [x, *args], kwargs)


def var_mean(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Var_mean operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Var_mean output.
    """
    return record_op("Var_mean", [x, *args], kwargs)


def vdot(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Vdot operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Vdot output.
    """
    return record_op("Vdot", [x, *args], kwargs)


def view_as_complex(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the View_as_complex operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the View_as_complex output.
    """
    return record_op("View_as_complex", [x, *args], kwargs)


def view_as_complex_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the View_as_complex_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the View_as_complex_copy output.
    """
    return record_op("View_as_complex_copy", [x, *args], kwargs)


def view_as_real(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the View_as_real operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the View_as_real output.
    """
    return record_op("View_as_real", [x, *args], kwargs)


def view_as_real_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the View_as_real_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the View_as_real_copy output.
    """
    return record_op("View_as_real_copy", [x, *args], kwargs)


def view_copy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the View_copy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the View_copy output.
    """
    return record_op("View_copy", [x, *args], kwargs)


def vsplit(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Vsplit operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Vsplit output.
    """
    return record_op("Vsplit", [x, *args], kwargs)


def vstack(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Vstack operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Vstack output.
    """
    return record_op("Vstack", [x, *args], kwargs)


def xlogy(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Xlogy operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Xlogy output.
    """
    return record_op("Xlogy", [x, *args], kwargs)


def xlogy_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Xlogy_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Xlogy_ output.
    """
    return record_op("Xlogy_", [x, *args], kwargs)


def zero_(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Zero_ operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Zero_ output.
    """
    return record_op("Zero_", [x, *args], kwargs)


def zeros_like(
    x: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Record the Zeros_like operation on the input tensor.

    Args:
        x: Primary input tensor.
        *args: Additional positional tensor or scalar inputs.
        **kwargs: Additional keyword attributes for the operation.

    Returns:
        A symbolic Tensor representing the Zeros_like output.
    """
    return record_op("Zeros_like", [x, *args], kwargs)
