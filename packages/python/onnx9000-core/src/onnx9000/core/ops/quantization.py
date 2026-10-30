"""Quantization and integer arithmetic ONNX operators."""

from onnx9000.core.ir import AttributeValue, Tensor
from onnx9000.core.ops.base import record_op
from onnx9000.core.registry import register_op


@register_op("onnx9000.custom", "FakeQuantize")
def fake_quantize(x: Tensor, scale: Tensor, zero_point: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute Fake Quantization."""
    return record_op("FakeQuantize", [x, scale, zero_point], kwargs)


@register_op("ai.onnx", "ConvInteger")
def conv_integer(
    x: Tensor,
    w: Tensor,
    x_zero_point: Tensor = None,
    w_zero_point: Tensor = None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Compute ConvInteger."""
    inputs = [x, w]
    if x_zero_point is not None:
        inputs.append(x_zero_point)
    elif w_zero_point is not None:
        inputs.append(record_op("Constant", [], {"value": []}))
    if w_zero_point is not None:
        inputs.append(w_zero_point)
    return record_op("ConvInteger", inputs, kwargs)


@register_op("ai.onnx", "DynamicQuantizeLinear")
def dynamic_quantize_linear(x: Tensor) -> Tensor:
    """Compute DynamicQuantizeLinear."""
    return record_op("DynamicQuantizeLinear", [x])


@register_op("ai.onnx", "BitCast")
def bit_cast(x: Tensor) -> Tensor:
    """Compute BitCast."""
    return record_op("BitCast", [x])


@register_op("ai.onnx", "MatMulInteger")
def mat_mul_integer(x: Tensor) -> Tensor:
    """Compute MatMulInteger."""
    return record_op("MatMulInteger", [x])


@register_op("ai.onnx", "QLinearConv")
def q_linear_conv(x: Tensor) -> Tensor:
    """Compute QLinearConv."""
    return record_op("QLinearConv", [x])


@register_op("ai.onnx", "QLinearMatMul")
def q_linear_mat_mul(x: Tensor) -> Tensor:
    """Compute QLinearMatMul."""
    return record_op("QLinearMatMul", [x])


@register_op("ai.onnx", "QuantizeLinear")
def quantize_linear(
    x: Tensor, y_scale: Tensor, y_zero_point: Tensor = None, **kwargs: AttributeValue
) -> Tensor:
    """Compute QuantizeLinear."""
    return record_op("QuantizeLinear", [x, y_scale] + ([y_zero_point] if y_zero_point else []))


@register_op("ai.onnx", "DequantizeLinear")
def dequantize_linear(
    x: Tensor, x_scale: Tensor, x_zero_point: Tensor = None, **kwargs: AttributeValue
) -> Tensor:
    """Compute DequantizeLinear."""
    return record_op("DequantizeLinear", [x, x_scale] + ([x_zero_point] if x_zero_point else []))
