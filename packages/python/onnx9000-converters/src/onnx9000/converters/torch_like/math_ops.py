"""PyTorch-compatible elementwise mathematical and trigonometric operations."""

from onnx9000.core.ir import AttributeValue, Tensor


def abs(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply abs to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for abs.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Abs", [input_tensor, *args], kwargs)


def acos(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply acos to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for acos.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Acos", [input_tensor, *args], kwargs)


def acosh(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply acosh to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for acosh.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Acosh", [input_tensor, *args], kwargs)


def add(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply add to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for add.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Add", [input_tensor, *args], kwargs)


def asin(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply asin to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for asin.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Asin", [input_tensor, *args], kwargs)


def asinh(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply asinh to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for asinh.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Asinh", [input_tensor, *args], kwargs)


def atan(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply atan to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for atan.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Atan", [input_tensor, *args], kwargs)


def atanh(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply atanh to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for atanh.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Atanh", [input_tensor, *args], kwargs)


def bitwise_and(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply bitwise_and to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for bitwise_and.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Bitwise_and", [input_tensor, *args], kwargs)


def bitwise_not(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply bitwise_not to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for bitwise_not.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Bitwise_not", [input_tensor, *args], kwargs)


def bitwise_or(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply bitwise_or to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for bitwise_or.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Bitwise_or", [input_tensor, *args], kwargs)


def bitwise_xor(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply bitwise_xor to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for bitwise_xor.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Bitwise_xor", [input_tensor, *args], kwargs)


def ceil(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply ceil to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for ceil.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Ceil", [input_tensor, *args], kwargs)


def cos(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cos to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cos.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Cos", [input_tensor, *args], kwargs)


def cosh(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cosh to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cosh.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Cosh", [input_tensor, *args], kwargs)


def div(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply div to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for div.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Div", [input_tensor, *args], kwargs)


def equal(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply equal to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for equal.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Equal", [input_tensor, *args], kwargs)


def erf(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply erf to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for erf.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Erf", [input_tensor, *args], kwargs)


def exp(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply exp to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for exp.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Exp", [input_tensor, *args], kwargs)


def floor(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply floor to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for floor.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Floor", [input_tensor, *args], kwargs)


def greater(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply greater to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for greater.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Greater", [input_tensor, *args], kwargs)


def isinf(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply isinf to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for isinf.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Isinf", [input_tensor, *args], kwargs)


def isnan(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply isnan to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for isnan.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Isnan", [input_tensor, *args], kwargs)


def less(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply less to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for less.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Less", [input_tensor, *args], kwargs)


def mul(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply mul to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for mul.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Mul", [input_tensor, *args], kwargs)


def neg(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply neg to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for neg.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Neg", [input_tensor, *args], kwargs)


def pow(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply pow to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for pow.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Pow", [input_tensor, *args], kwargs)


def reciprocal(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply reciprocal to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for reciprocal.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Reciprocal", [input_tensor, *args], kwargs)


def round(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply round to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for round.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Round", [input_tensor, *args], kwargs)


def sign(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply sign to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for sign.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Sign", [input_tensor, *args], kwargs)


def sin(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply sin to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for sin.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Sin", [input_tensor, *args], kwargs)


def sinh(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply sinh to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for sinh.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Sinh", [input_tensor, *args], kwargs)


def sub(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply sub to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for sub.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Sub", [input_tensor, *args], kwargs)


def tan(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply tan to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for tan.
    """
    from onnx9000.core.ops.base import record_op

    return record_op("Tan", [input_tensor, *args], kwargs)


def abs_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply abs_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for abs_.
    """
    from onnx9000.core.ops.torch_auto import abs_ as core_abs_

    return core_abs_(input_tensor, *args, **kwargs)


def absolute(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply absolute to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for absolute.
    """
    from onnx9000.core.ops.torch_auto import absolute as core_absolute

    return core_absolute(input_tensor, *args, **kwargs)


def acos_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply acos_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for acos_.
    """
    from onnx9000.core.ops.torch_auto import acos_ as core_acos_

    return core_acos_(input_tensor, *args, **kwargs)


def acosh_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply acosh_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for acosh_.
    """
    from onnx9000.core.ops.torch_auto import acosh_ as core_acosh_

    return core_acosh_(input_tensor, *args, **kwargs)


def addcdiv(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply addcdiv to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for addcdiv.
    """
    from onnx9000.core.ops.torch_auto import addcdiv as core_addcdiv

    return core_addcdiv(input_tensor, *args, **kwargs)


def addcmul(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply addcmul to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for addcmul.
    """
    from onnx9000.core.ops.torch_auto import addcmul as core_addcmul

    return core_addcmul(input_tensor, *args, **kwargs)


def angle(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply angle to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for angle.
    """
    from onnx9000.core.ops.torch_auto import angle as core_angle

    return core_angle(input_tensor, *args, **kwargs)


def arccos(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply arccos to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for arccos.
    """
    from onnx9000.core.ops.torch_auto import arccos as core_arccos

    return core_arccos(input_tensor, *args, **kwargs)


def arccos_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply arccos_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for arccos_.
    """
    from onnx9000.core.ops.torch_auto import arccos_ as core_arccos_

    return core_arccos_(input_tensor, *args, **kwargs)


def arccosh(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply arccosh to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for arccosh.
    """
    from onnx9000.core.ops.torch_auto import arccosh as core_arccosh

    return core_arccosh(input_tensor, *args, **kwargs)


def arccosh_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply arccosh_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for arccosh_.
    """
    from onnx9000.core.ops.torch_auto import arccosh_ as core_arccosh_

    return core_arccosh_(input_tensor, *args, **kwargs)


def arcsin(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply arcsin to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for arcsin.
    """
    from onnx9000.core.ops.torch_auto import arcsin as core_arcsin

    return core_arcsin(input_tensor, *args, **kwargs)


def arcsin_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply arcsin_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for arcsin_.
    """
    from onnx9000.core.ops.torch_auto import arcsin_ as core_arcsin_

    return core_arcsin_(input_tensor, *args, **kwargs)


def arcsinh(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply arcsinh to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for arcsinh.
    """
    from onnx9000.core.ops.torch_auto import arcsinh as core_arcsinh

    return core_arcsinh(input_tensor, *args, **kwargs)


def arcsinh_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply arcsinh_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for arcsinh_.
    """
    from onnx9000.core.ops.torch_auto import arcsinh_ as core_arcsinh_

    return core_arcsinh_(input_tensor, *args, **kwargs)


def arctan(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply arctan to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for arctan.
    """
    from onnx9000.core.ops.torch_auto import arctan as core_arctan

    return core_arctan(input_tensor, *args, **kwargs)


def arctan2(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply arctan2 to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for arctan2.
    """
    from onnx9000.core.ops.torch_auto import arctan2 as core_arctan2

    return core_arctan2(input_tensor, *args, **kwargs)


def arctan_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply arctan_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for arctan_.
    """
    from onnx9000.core.ops.torch_auto import arctan_ as core_arctan_

    return core_arctan_(input_tensor, *args, **kwargs)


def arctanh(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply arctanh to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for arctanh.
    """
    from onnx9000.core.ops.torch_auto import arctanh as core_arctanh

    return core_arctanh(input_tensor, *args, **kwargs)


def arctanh_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply arctanh_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for arctanh_.
    """
    from onnx9000.core.ops.torch_auto import arctanh_ as core_arctanh_

    return core_arctanh_(input_tensor, *args, **kwargs)


def asin_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply asin_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for asin_.
    """
    from onnx9000.core.ops.torch_auto import asin_ as core_asin_

    return core_asin_(input_tensor, *args, **kwargs)


def asinh_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply asinh_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for asinh_.
    """
    from onnx9000.core.ops.torch_auto import asinh_ as core_asinh_

    return core_asinh_(input_tensor, *args, **kwargs)


def atan2(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply atan2 to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for atan2.
    """
    from onnx9000.core.ops.torch_auto import atan2 as core_atan2

    return core_atan2(input_tensor, *args, **kwargs)


def atan_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply atan_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for atan_.
    """
    from onnx9000.core.ops.torch_auto import atan_ as core_atan_

    return core_atan_(input_tensor, *args, **kwargs)


def atanh_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply atanh_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for atanh_.
    """
    from onnx9000.core.ops.torch_auto import atanh_ as core_atanh_

    return core_atanh_(input_tensor, *args, **kwargs)


def bitwise_left_shift(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply bitwise_left_shift to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for bitwise_left_shift.
    """
    from onnx9000.core.ops.torch_auto import bitwise_left_shift as core_bitwise_left_shift

    return core_bitwise_left_shift(input_tensor, *args, **kwargs)


def bitwise_right_shift(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply bitwise_right_shift to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for bitwise_right_shift.
    """
    from onnx9000.core.ops.torch_auto import bitwise_right_shift as core_bitwise_right_shift

    return core_bitwise_right_shift(input_tensor, *args, **kwargs)


def cdist(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cdist to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cdist.
    """
    from onnx9000.core.ops.torch_auto import cdist as core_cdist

    return core_cdist(input_tensor, *args, **kwargs)


def ceil_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply ceil_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for ceil_.
    """
    from onnx9000.core.ops.torch_auto import ceil_ as core_ceil_

    return core_ceil_(input_tensor, *args, **kwargs)


def clamp(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply clamp to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for clamp.
    """
    from onnx9000.core.ops.torch_auto import clamp as core_clamp

    return core_clamp(input_tensor, *args, **kwargs)


def clamp_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply clamp_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for clamp_.
    """
    from onnx9000.core.ops.torch_auto import clamp_ as core_clamp_

    return core_clamp_(input_tensor, *args, **kwargs)


def clamp_max(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply clamp_max to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for clamp_max.
    """
    from onnx9000.core.ops.torch_auto import clamp_max as core_clamp_max

    return core_clamp_max(input_tensor, *args, **kwargs)


def clamp_max_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply clamp_max_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for clamp_max_.
    """
    from onnx9000.core.ops.torch_auto import clamp_max_ as core_clamp_max_

    return core_clamp_max_(input_tensor, *args, **kwargs)


def clamp_min(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply clamp_min to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for clamp_min.
    """
    from onnx9000.core.ops.torch_auto import clamp_min as core_clamp_min

    return core_clamp_min(input_tensor, *args, **kwargs)


def clamp_min_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply clamp_min_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for clamp_min_.
    """
    from onnx9000.core.ops.torch_auto import clamp_min_ as core_clamp_min_

    return core_clamp_min_(input_tensor, *args, **kwargs)


def clip_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply clip_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for clip_.
    """
    from onnx9000.core.ops.torch_auto import clip_ as core_clip_

    return core_clip_(input_tensor, *args, **kwargs)


def complex(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply complex to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for complex.
    """
    from onnx9000.core.ops.torch_auto import complex as core_complex

    return core_complex(input_tensor, *args, **kwargs)


def conj(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply conj to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for conj.
    """
    from onnx9000.core.ops.torch_auto import conj as core_conj

    return core_conj(input_tensor, *args, **kwargs)


def conj_physical(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply conj_physical to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for conj_physical.
    """
    from onnx9000.core.ops.torch_auto import conj_physical as core_conj_physical

    return core_conj_physical(input_tensor, *args, **kwargs)


def conj_physical_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply conj_physical_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for conj_physical_.
    """
    from onnx9000.core.ops.torch_auto import conj_physical_ as core_conj_physical_

    return core_conj_physical_(input_tensor, *args, **kwargs)


def copysign(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply copysign to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for copysign.
    """
    from onnx9000.core.ops.torch_auto import copysign as core_copysign

    return core_copysign(input_tensor, *args, **kwargs)


def cos_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cos_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cos_.
    """
    from onnx9000.core.ops.torch_auto import cos_ as core_cos_

    return core_cos_(input_tensor, *args, **kwargs)


def cosh_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply cosh_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for cosh_.
    """
    from onnx9000.core.ops.torch_auto import cosh_ as core_cosh_

    return core_cosh_(input_tensor, *args, **kwargs)


def deg2rad(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply deg2rad to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for deg2rad.
    """
    from onnx9000.core.ops.torch_auto import deg2rad as core_deg2rad

    return core_deg2rad(input_tensor, *args, **kwargs)


def deg2rad_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply deg2rad_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for deg2rad_.
    """
    from onnx9000.core.ops.torch_auto import deg2rad_ as core_deg2rad_

    return core_deg2rad_(input_tensor, *args, **kwargs)


def diff(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply diff to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for diff.
    """
    from onnx9000.core.ops.torch_auto import diff as core_diff

    return core_diff(input_tensor, *args, **kwargs)


def digamma(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply digamma to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for digamma.
    """
    from onnx9000.core.ops.torch_auto import digamma as core_digamma

    return core_digamma(input_tensor, *args, **kwargs)


def divide(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply divide to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for divide.
    """
    from onnx9000.core.ops.torch_auto import divide as core_divide

    return core_divide(input_tensor, *args, **kwargs)


def eq(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply eq to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for eq.
    """
    from onnx9000.core.ops.torch_auto import eq as core_eq

    return core_eq(input_tensor, *args, **kwargs)


def erf_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply erf_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for erf_.
    """
    from onnx9000.core.ops.torch_auto import erf_ as core_erf_

    return core_erf_(input_tensor, *args, **kwargs)


def erfc(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply erfc to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for erfc.
    """
    from onnx9000.core.ops.torch_auto import erfc as core_erfc

    return core_erfc(input_tensor, *args, **kwargs)


def erfc_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply erfc_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for erfc_.
    """
    from onnx9000.core.ops.torch_auto import erfc_ as core_erfc_

    return core_erfc_(input_tensor, *args, **kwargs)


def erfinv(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply erfinv to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for erfinv.
    """
    from onnx9000.core.ops.torch_auto import erfinv as core_erfinv

    return core_erfinv(input_tensor, *args, **kwargs)


def exp2(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply exp2 to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for exp2.
    """
    from onnx9000.core.ops.torch_auto import exp2 as core_exp2

    return core_exp2(input_tensor, *args, **kwargs)


def exp2_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply exp2_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for exp2_.
    """
    from onnx9000.core.ops.torch_auto import exp2_ as core_exp2_

    return core_exp2_(input_tensor, *args, **kwargs)


def exp_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply exp_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for exp_.
    """
    from onnx9000.core.ops.torch_auto import exp_ as core_exp_

    return core_exp_(input_tensor, *args, **kwargs)


def expm1_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply expm1_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for expm1_.
    """
    from onnx9000.core.ops.torch_auto import expm1_ as core_expm1_

    return core_expm1_(input_tensor, *args, **kwargs)


def fix(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply fix to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for fix.
    """
    from onnx9000.core.ops.torch_auto import fix as core_fix

    return core_fix(input_tensor, *args, **kwargs)


def fix_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply fix_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for fix_.
    """
    from onnx9000.core.ops.torch_auto import fix_ as core_fix_

    return core_fix_(input_tensor, *args, **kwargs)


def float_power(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply float_power to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for float_power.
    """
    from onnx9000.core.ops.torch_auto import float_power as core_float_power

    return core_float_power(input_tensor, *args, **kwargs)


def floor_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply floor_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for floor_.
    """
    from onnx9000.core.ops.torch_auto import floor_ as core_floor_

    return core_floor_(input_tensor, *args, **kwargs)


def floor_divide(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply floor_divide to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for floor_divide.
    """
    from onnx9000.core.ops.torch_auto import floor_divide as core_floor_divide

    return core_floor_divide(input_tensor, *args, **kwargs)


def fmax(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply fmax to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for fmax.
    """
    from onnx9000.core.ops.torch_auto import fmax as core_fmax

    return core_fmax(input_tensor, *args, **kwargs)


def fmin(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply fmin to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for fmin.
    """
    from onnx9000.core.ops.torch_auto import fmin as core_fmin

    return core_fmin(input_tensor, *args, **kwargs)


def frac(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply frac to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for frac.
    """
    from onnx9000.core.ops.torch_auto import frac as core_frac

    return core_frac(input_tensor, *args, **kwargs)


def frac_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply frac_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for frac_.
    """
    from onnx9000.core.ops.torch_auto import frac_ as core_frac_

    return core_frac_(input_tensor, *args, **kwargs)


def frexp(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply frexp to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for frexp.
    """
    from onnx9000.core.ops.torch_auto import frexp as core_frexp

    return core_frexp(input_tensor, *args, **kwargs)


def gcd(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply gcd to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for gcd.
    """
    from onnx9000.core.ops.torch_auto import gcd as core_gcd

    return core_gcd(input_tensor, *args, **kwargs)


def gcd_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply gcd_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for gcd_.
    """
    from onnx9000.core.ops.torch_auto import gcd_ as core_gcd_

    return core_gcd_(input_tensor, *args, **kwargs)


def ge(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply ge to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for ge.
    """
    from onnx9000.core.ops.torch_auto import ge as core_ge

    return core_ge(input_tensor, *args, **kwargs)


def gradient(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply gradient to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for gradient.
    """
    from onnx9000.core.ops.torch_auto import gradient as core_gradient

    return core_gradient(input_tensor, *args, **kwargs)


def greater_equal(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply greater_equal to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for greater_equal.
    """
    from onnx9000.core.ops.torch_auto import greater_equal as core_greater_equal

    return core_greater_equal(input_tensor, *args, **kwargs)


def gt(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply gt to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for gt.
    """
    from onnx9000.core.ops.torch_auto import gt as core_gt

    return core_gt(input_tensor, *args, **kwargs)


def hash_tensor(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply hash_tensor to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for hash_tensor.
    """
    from onnx9000.core.ops.torch_auto import hash_tensor as core_hash_tensor

    return core_hash_tensor(input_tensor, *args, **kwargs)


def heaviside(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply heaviside to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for heaviside.
    """
    from onnx9000.core.ops.torch_auto import heaviside as core_heaviside

    return core_heaviside(input_tensor, *args, **kwargs)


def hypot(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply hypot to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for hypot.
    """
    from onnx9000.core.ops.torch_auto import hypot as core_hypot

    return core_hypot(input_tensor, *args, **kwargs)


def i0(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply i0 to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for i0.
    """
    from onnx9000.core.ops.torch_auto import i0 as core_i0

    return core_i0(input_tensor, *args, **kwargs)


def i0_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply i0_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for i0_.
    """
    from onnx9000.core.ops.torch_auto import i0_ as core_i0_

    return core_i0_(input_tensor, *args, **kwargs)


def igamma(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply igamma to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for igamma.
    """
    from onnx9000.core.ops.torch_auto import igamma as core_igamma

    return core_igamma(input_tensor, *args, **kwargs)


def igammac(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply igammac to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for igammac.
    """
    from onnx9000.core.ops.torch_auto import igammac as core_igammac

    return core_igammac(input_tensor, *args, **kwargs)


def imag(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply imag to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for imag.
    """
    from onnx9000.core.ops.torch_auto import imag as core_imag

    return core_imag(input_tensor, *args, **kwargs)


def is_complex(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply is_complex to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for is_complex.
    """
    from onnx9000.core.ops.torch_auto import is_complex as core_is_complex

    return core_is_complex(input_tensor, *args, **kwargs)


def is_conj(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply is_conj to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for is_conj.
    """
    from onnx9000.core.ops.torch_auto import is_conj as core_is_conj

    return core_is_conj(input_tensor, *args, **kwargs)


def is_floating_point(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply is_floating_point to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for is_floating_point.
    """
    from onnx9000.core.ops.torch_auto import is_floating_point as core_is_floating_point

    return core_is_floating_point(input_tensor, *args, **kwargs)


def is_neg(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply is_neg to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for is_neg.
    """
    from onnx9000.core.ops.torch_auto import is_neg as core_is_neg

    return core_is_neg(input_tensor, *args, **kwargs)


def is_nonzero(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply is_nonzero to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for is_nonzero.
    """
    from onnx9000.core.ops.torch_auto import is_nonzero as core_is_nonzero

    return core_is_nonzero(input_tensor, *args, **kwargs)


def is_same_size(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply is_same_size to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for is_same_size.
    """
    from onnx9000.core.ops.torch_auto import is_same_size as core_is_same_size

    return core_is_same_size(input_tensor, *args, **kwargs)


def is_signed(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply is_signed to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for is_signed.
    """
    from onnx9000.core.ops.torch_auto import is_signed as core_is_signed

    return core_is_signed(input_tensor, *args, **kwargs)


def isclose(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply isclose to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for isclose.
    """
    from onnx9000.core.ops.torch_auto import isclose as core_isclose

    return core_isclose(input_tensor, *args, **kwargs)


def isneginf(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply isneginf to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for isneginf.
    """
    from onnx9000.core.ops.torch_auto import isneginf as core_isneginf

    return core_isneginf(input_tensor, *args, **kwargs)


def isposinf(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply isposinf to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for isposinf.
    """
    from onnx9000.core.ops.torch_auto import isposinf as core_isposinf

    return core_isposinf(input_tensor, *args, **kwargs)


def isreal(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply isreal to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for isreal.
    """
    from onnx9000.core.ops.torch_auto import isreal as core_isreal

    return core_isreal(input_tensor, *args, **kwargs)


def lcm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply lcm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for lcm.
    """
    from onnx9000.core.ops.torch_auto import lcm as core_lcm

    return core_lcm(input_tensor, *args, **kwargs)


def lcm_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply lcm_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for lcm_.
    """
    from onnx9000.core.ops.torch_auto import lcm_ as core_lcm_

    return core_lcm_(input_tensor, *args, **kwargs)


def ldexp(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply ldexp to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for ldexp.
    """
    from onnx9000.core.ops.torch_auto import ldexp as core_ldexp

    return core_ldexp(input_tensor, *args, **kwargs)


def ldexp_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply ldexp_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for ldexp_.
    """
    from onnx9000.core.ops.torch_auto import ldexp_ as core_ldexp_

    return core_ldexp_(input_tensor, *args, **kwargs)


def le(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply le to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for le.
    """
    from onnx9000.core.ops.torch_auto import le as core_le

    return core_le(input_tensor, *args, **kwargs)


def lerp(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply lerp to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for lerp.
    """
    from onnx9000.core.ops.torch_auto import lerp as core_lerp

    return core_lerp(input_tensor, *args, **kwargs)


def less_equal(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply less_equal to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for less_equal.
    """
    from onnx9000.core.ops.torch_auto import less_equal as core_less_equal

    return core_less_equal(input_tensor, *args, **kwargs)


def lgamma(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply lgamma to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for lgamma.
    """
    from onnx9000.core.ops.torch_auto import lgamma as core_lgamma

    return core_lgamma(input_tensor, *args, **kwargs)


def log10_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply log10_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for log10_.
    """
    from onnx9000.core.ops.torch_auto import log10_ as core_log10_

    return core_log10_(input_tensor, *args, **kwargs)


def log1p_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply log1p_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for log1p_.
    """
    from onnx9000.core.ops.torch_auto import log1p_ as core_log1p_

    return core_log1p_(input_tensor, *args, **kwargs)


def log2_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply log2_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for log2_.
    """
    from onnx9000.core.ops.torch_auto import log2_ as core_log2_

    return core_log2_(input_tensor, *args, **kwargs)


def log_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply log_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for log_.
    """
    from onnx9000.core.ops.torch_auto import log_ as core_log_

    return core_log_(input_tensor, *args, **kwargs)


def logaddexp(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply logaddexp to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for logaddexp.
    """
    from onnx9000.core.ops.torch_auto import logaddexp as core_logaddexp

    return core_logaddexp(input_tensor, *args, **kwargs)


def logaddexp2(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply logaddexp2 to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for logaddexp2.
    """
    from onnx9000.core.ops.torch_auto import logaddexp2 as core_logaddexp2

    return core_logaddexp2(input_tensor, *args, **kwargs)


def logit(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply logit to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for logit.
    """
    from onnx9000.core.ops.torch_auto import logit as core_logit

    return core_logit(input_tensor, *args, **kwargs)


def logit_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply logit_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for logit_.
    """
    from onnx9000.core.ops.torch_auto import logit_ as core_logit_

    return core_logit_(input_tensor, *args, **kwargs)


def lt(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply lt to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for lt.
    """
    from onnx9000.core.ops.torch_auto import lt as core_lt

    return core_lt(input_tensor, *args, **kwargs)


def multiply(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply multiply to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for multiply.
    """
    from onnx9000.core.ops.torch_auto import multiply as core_multiply

    return core_multiply(input_tensor, *args, **kwargs)


def mvlgamma(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply mvlgamma to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for mvlgamma.
    """
    from onnx9000.core.ops.torch_auto import mvlgamma as core_mvlgamma

    return core_mvlgamma(input_tensor, *args, **kwargs)


def nan_to_num(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply nan_to_num to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for nan_to_num.
    """
    from onnx9000.core.ops.torch_auto import nan_to_num as core_nan_to_num

    return core_nan_to_num(input_tensor, *args, **kwargs)


def nan_to_num_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply nan_to_num_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for nan_to_num_.
    """
    from onnx9000.core.ops.torch_auto import nan_to_num_ as core_nan_to_num_

    return core_nan_to_num_(input_tensor, *args, **kwargs)


def ne(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply ne to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for ne.
    """
    from onnx9000.core.ops.torch_auto import ne as core_ne

    return core_ne(input_tensor, *args, **kwargs)


def neg_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply neg_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for neg_.
    """
    from onnx9000.core.ops.torch_auto import neg_ as core_neg_

    return core_neg_(input_tensor, *args, **kwargs)


def negative(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply negative to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for negative.
    """
    from onnx9000.core.ops.torch_auto import negative as core_negative

    return core_negative(input_tensor, *args, **kwargs)


def negative_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply negative_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for negative_.
    """
    from onnx9000.core.ops.torch_auto import negative_ as core_negative_

    return core_negative_(input_tensor, *args, **kwargs)


def nextafter(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply nextafter to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for nextafter.
    """
    from onnx9000.core.ops.torch_auto import nextafter as core_nextafter

    return core_nextafter(input_tensor, *args, **kwargs)


def not_equal(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply not_equal to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for not_equal.
    """
    from onnx9000.core.ops.torch_auto import not_equal as core_not_equal

    return core_not_equal(input_tensor, *args, **kwargs)


def polar(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply polar to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for polar.
    """
    from onnx9000.core.ops.torch_auto import polar as core_polar

    return core_polar(input_tensor, *args, **kwargs)


def polygamma(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply polygamma to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for polygamma.
    """
    from onnx9000.core.ops.torch_auto import polygamma as core_polygamma

    return core_polygamma(input_tensor, *args, **kwargs)


def positive(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply positive to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for positive.
    """
    from onnx9000.core.ops.torch_auto import positive as core_positive

    return core_positive(input_tensor, *args, **kwargs)


def rad2deg(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply rad2deg to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for rad2deg.
    """
    from onnx9000.core.ops.torch_auto import rad2deg as core_rad2deg

    return core_rad2deg(input_tensor, *args, **kwargs)


def rad2deg_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply rad2deg_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for rad2deg_.
    """
    from onnx9000.core.ops.torch_auto import rad2deg_ as core_rad2deg_

    return core_rad2deg_(input_tensor, *args, **kwargs)


def real(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply real to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for real.
    """
    from onnx9000.core.ops.torch_auto import real as core_real

    return core_real(input_tensor, *args, **kwargs)


def reciprocal_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply reciprocal_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for reciprocal_.
    """
    from onnx9000.core.ops.torch_auto import reciprocal_ as core_reciprocal_

    return core_reciprocal_(input_tensor, *args, **kwargs)


def remainder(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply remainder to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for remainder.
    """
    from onnx9000.core.ops.torch_auto import remainder as core_remainder

    return core_remainder(input_tensor, *args, **kwargs)


def renorm(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply renorm to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for renorm.
    """
    from onnx9000.core.ops.torch_auto import renorm as core_renorm

    return core_renorm(input_tensor, *args, **kwargs)


def resolve_conj(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply resolve_conj to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for resolve_conj.
    """
    from onnx9000.core.ops.torch_auto import resolve_conj as core_resolve_conj

    return core_resolve_conj(input_tensor, *args, **kwargs)


def resolve_neg(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply resolve_neg to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for resolve_neg.
    """
    from onnx9000.core.ops.torch_auto import resolve_neg as core_resolve_neg

    return core_resolve_neg(input_tensor, *args, **kwargs)


def round_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply round_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for round_.
    """
    from onnx9000.core.ops.torch_auto import round_ as core_round_

    return core_round_(input_tensor, *args, **kwargs)


def rsqrt(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply rsqrt to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for rsqrt.
    """
    from onnx9000.core.ops.torch_auto import rsqrt as core_rsqrt

    return core_rsqrt(input_tensor, *args, **kwargs)


def rsqrt_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply rsqrt_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for rsqrt_.
    """
    from onnx9000.core.ops.torch_auto import rsqrt_ as core_rsqrt_

    return core_rsqrt_(input_tensor, *args, **kwargs)


def rsub(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply rsub to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for rsub.
    """
    from onnx9000.core.ops.torch_auto import rsub as core_rsub

    return core_rsub(input_tensor, *args, **kwargs)


def sgn(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply sgn to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for sgn.
    """
    from onnx9000.core.ops.torch_auto import sgn as core_sgn

    return core_sgn(input_tensor, *args, **kwargs)


def signbit(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply signbit to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for signbit.
    """
    from onnx9000.core.ops.torch_auto import signbit as core_signbit

    return core_signbit(input_tensor, *args, **kwargs)


def sin_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply sin_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for sin_.
    """
    from onnx9000.core.ops.torch_auto import sin_ as core_sin_

    return core_sin_(input_tensor, *args, **kwargs)


def sinc(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply sinc to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for sinc.
    """
    from onnx9000.core.ops.torch_auto import sinc as core_sinc

    return core_sinc(input_tensor, *args, **kwargs)


def sinc_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply sinc_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for sinc_.
    """
    from onnx9000.core.ops.torch_auto import sinc_ as core_sinc_

    return core_sinc_(input_tensor, *args, **kwargs)


def sinh_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply sinh_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for sinh_.
    """
    from onnx9000.core.ops.torch_auto import sinh_ as core_sinh_

    return core_sinh_(input_tensor, *args, **kwargs)


def sqrt_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply sqrt_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for sqrt_.
    """
    from onnx9000.core.ops.torch_auto import sqrt_ as core_sqrt_

    return core_sqrt_(input_tensor, *args, **kwargs)


def square(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply square to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for square.
    """
    from onnx9000.core.ops.torch_auto import square as core_square

    return core_square(input_tensor, *args, **kwargs)


def square_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply square_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for square_.
    """
    from onnx9000.core.ops.torch_auto import square_ as core_square_

    return core_square_(input_tensor, *args, **kwargs)


def subtract(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply subtract to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for subtract.
    """
    from onnx9000.core.ops.torch_auto import subtract as core_subtract

    return core_subtract(input_tensor, *args, **kwargs)


def tan_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply tan_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for tan_.
    """
    from onnx9000.core.ops.torch_auto import tan_ as core_tan_

    return core_tan_(input_tensor, *args, **kwargs)


def true_divide(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply true_divide to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for true_divide.
    """
    from onnx9000.core.ops.torch_auto import true_divide as core_true_divide

    return core_true_divide(input_tensor, *args, **kwargs)


def trunc(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply trunc to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for trunc.
    """
    from onnx9000.core.ops.torch_auto import trunc as core_trunc

    return core_trunc(input_tensor, *args, **kwargs)


def trunc_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply trunc_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for trunc_.
    """
    from onnx9000.core.ops.torch_auto import trunc_ as core_trunc_

    return core_trunc_(input_tensor, *args, **kwargs)


def xlogy(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply xlogy to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for xlogy.
    """
    from onnx9000.core.ops.torch_auto import xlogy as core_xlogy

    return core_xlogy(input_tensor, *args, **kwargs)


def xlogy_(
    input_tensor: Tensor,
    *args: Tensor | int | float | str | bool | None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Apply xlogy_ to the input tensor.

    Args:
        input_tensor: Primary input Tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword attributes.

    Returns:
        Symbolic output Tensor for xlogy_.
    """
    from onnx9000.core.ops.torch_auto import xlogy_ as core_xlogy_

    return core_xlogy_(input_tensor, *args, **kwargs)
