"""Mathematical and elementwise ONNX operators."""

from onnx9000.core.ir import AttributeValue, Tensor
from onnx9000.core.ops.base import record_op
from onnx9000.core.registry import register_op


@register_op("ai.onnx", "Abs")
def abs(x: Tensor) -> Tensor:
    """Compute Absolute Value."""
    return record_op("Abs", [x])


@register_op("ai.onnx", "Acos")
def acos(x: Tensor) -> Tensor:
    """Compute Inverse Cosine."""
    return record_op("Acos", [x])


@register_op("ai.onnx", "Acosh")
def acosh(x: Tensor) -> Tensor:
    """Compute Inverse Hyperbolic Cosine."""
    return record_op("Acosh", [x])


@register_op("ai.onnx", "Asin")
def asin(x: Tensor) -> Tensor:
    """Compute Inverse Sine."""
    return record_op("Asin", [x])


@register_op("ai.onnx", "Asinh")
def asinh(x: Tensor) -> Tensor:
    """Compute Inverse Hyperbolic Sine."""
    return record_op("Asinh", [x])


@register_op("ai.onnx", "Atan")
def atan(x: Tensor) -> Tensor:
    """Compute Inverse Tangent."""
    return record_op("Atan", [x])


@register_op("ai.onnx", "Atanh")
def atanh(x: Tensor) -> Tensor:
    """Compute Inverse Hyperbolic Tangent."""
    return record_op("Atanh", [x])


@register_op("ai.onnx", "Cos")
def cos(x: Tensor) -> Tensor:
    """Compute Cosine."""
    return record_op("Cos", [x])


@register_op("ai.onnx", "Cosh")
def cosh(x: Tensor) -> Tensor:
    """Compute Hyperbolic Cosine."""
    return record_op("Cosh", [x])


@register_op("ai.onnx", "Sin")
def sin(x: Tensor) -> Tensor:
    """Compute Sine."""
    return record_op("Sin", [x])


@register_op("ai.onnx", "Sinh")
def sinh(x: Tensor) -> Tensor:
    """Compute Hyperbolic Sine."""
    return record_op("Sinh", [x])


@register_op("ai.onnx", "Tan")
def tan(x: Tensor) -> Tensor:
    """Compute Tangent."""
    return record_op("Tan", [x])


@register_op("ai.onnx", "Max")
def max(x: Tensor, y: Tensor) -> Tensor:
    """Compute Max."""
    return record_op("Max", [x, y])


@register_op("ai.onnx", "Min")
def min(x: Tensor, y: Tensor) -> Tensor:
    """Compute Min."""
    return record_op("Min", [x, y])


@register_op("ai.onnx", "BitShift")
def bitshift(x: Tensor, y: Tensor, direction: str) -> Tensor:
    """Compute BitShift."""
    return record_op("BitShift", [x, y], {"direction": direction})


@register_op("ai.onnx", "BitwiseAnd")
def bitwise_and(x: Tensor, y: Tensor) -> Tensor:
    """Compute BitwiseAnd."""
    return record_op("BitwiseAnd", [x, y])


@register_op("ai.onnx", "BitwiseNot")
def bitwise_not(x: Tensor) -> Tensor:
    """Compute BitwiseNot."""
    return record_op("BitwiseNot", [x])


@register_op("ai.onnx", "BitwiseOr")
def bitwise_or(x: Tensor, y: Tensor) -> Tensor:
    """Compute BitwiseOr."""
    return record_op("BitwiseOr", [x, y])


@register_op("ai.onnx", "BitwiseXor")
def bitwise_xor(x: Tensor, y: Tensor) -> Tensor:
    """Compute BitwiseXor."""
    return record_op("BitwiseXor", [x, y])


@register_op("ai.onnx", "Ceil")
def ceil(x: Tensor) -> Tensor:
    """Compute Ceil."""
    return record_op("Ceil", [x])


@register_op("ai.onnx", "Floor")
def floor(x: Tensor) -> Tensor:
    """Compute Floor."""
    return record_op("Floor", [x])


@register_op("ai.onnx", "IsInf")
def isinf(x: Tensor) -> Tensor:
    """Compute IsInf."""
    return record_op("IsInf", [x])


@register_op("ai.onnx", "IsNaN")
def isnan(x: Tensor) -> Tensor:
    """Compute IsNaN."""
    return record_op("IsNaN", [x])


@register_op("ai.onnx", "Mod")
def mod(x: Tensor, y: Tensor) -> Tensor:
    """Compute Modulo."""
    return record_op("Mod", [x, y])


@register_op("ai.onnx", "Neg")
def neg(x: Tensor) -> Tensor:
    """Compute Negation."""
    return record_op("Neg", [x])


@register_op("ai.onnx", "Pow")
def pow(x: Tensor, y: Tensor) -> Tensor:
    """Compute Power."""
    return record_op("Pow", [x, y])


@register_op("ai.onnx", "Reciprocal")
def reciprocal(x: Tensor) -> Tensor:
    """Compute Reciprocal."""
    return record_op("Reciprocal", [x])


@register_op("ai.onnx", "Round")
def round(x: Tensor) -> Tensor:
    """Compute Round."""
    return record_op("Round", [x])


@register_op("ai.onnx", "Sign")
def sign(x: Tensor) -> Tensor:
    """Compute Sign."""
    return record_op("Sign", [x])


@register_op("ai.onnx", "BlackmanWindow")
def blackman_window(size: Tensor, output_datatype: int = 1, periodic: int = 1) -> Tensor:
    """Compute BlackmanWindow."""
    return record_op(
        "BlackmanWindow",
        [size],
        {"output_datatype": output_datatype, "periodic": periodic},
    )


@register_op("ai.onnx", "Det")
def det(x: Tensor) -> Tensor:
    """Compute Det."""
    return record_op("Det", [x])


@register_op("onnx9000.custom", "DCT")
def dct(x: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute Discrete Cosine Transform."""
    return record_op("DCT", [x], kwargs)


@register_op("ai.onnx", "LessOrEqual")
def less_or_equal(a: Tensor, b: Tensor) -> Tensor:
    """Compute LessOrEqual."""
    return record_op("LessOrEqual", [a, b])


@register_op("ai.onnx", "GreaterOrEqual")
def greater_or_equal(a: Tensor, b: Tensor) -> Tensor:
    """Compute GreaterOrEqual."""
    return record_op("GreaterOrEqual", [a, b])


@register_op("ai.onnx", "MelWeightMatrix")
def mel_weight_matrix(
    num_mel_bins: Tensor,
    dft_length: Tensor,
    sample_rate: Tensor,
    lower_edge_hertz: Tensor,
    upper_edge_hertz: Tensor,
    output_datatype: int = 1,
) -> Tensor:
    """Compute MelWeightMatrix."""
    inputs = [num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz]
    return record_op("MelWeightMatrix", inputs, {"output_datatype": output_datatype})


@register_op("ai.onnx", "Constant")
def clip(input: Tensor, min: Tensor = None, max: Tensor = None) -> Tensor:
    """Compute Clip."""
    inputs = [input]
    if min is not None:
        inputs.append(min)
    elif max is not None:
        inputs.append(record_op("Constant", [], {"value": []}))
    if max is not None:
        inputs.append(max)
    return record_op("Clip", inputs)


@register_op("ai.onnx", "Dft")
def dft(
    input: Tensor,
    dft_length: Tensor = None,
    axis: int = 1,
    inverse: int = 0,
    onesided: int = 0,
) -> Tensor:
    """Compute DFT."""
    inputs = [input]
    if dft_length is not None:
        inputs.append(dft_length)
    return record_op("DFT", inputs, {"axis": axis, "inverse": inverse, "onesided": onesided})


@register_op("ai.onnx", "Equal")
def equal(x: Tensor, y: Tensor) -> Tensor:
    """Compute Equal."""
    return record_op("Equal", [x, y])


@register_op("ai.onnx", "Greater")
def greater(x: Tensor, y: Tensor) -> Tensor:
    """Compute Greater."""
    return record_op("Greater", [x, y])


@register_op("ai.onnx", "Less")
def less(x: Tensor, y: Tensor) -> Tensor:
    """Compute Less."""
    return record_op("Less", [x, y])


@register_op("ai.onnx", "And")
def and_(x: Tensor, y: Tensor) -> Tensor:
    """Compute And."""
    return record_op("And", [x, y])


@register_op("ai.onnx", "Or")
def or_(x: Tensor, y: Tensor) -> Tensor:
    """Compute Or."""
    return record_op("Or", [x, y])


@register_op("ai.onnx", "Not")
def not_(x: Tensor) -> Tensor:
    """Compute Not."""
    return record_op("Not", [x])


@register_op("ai.onnx", "Add")
def add(x: Tensor, y: Tensor) -> Tensor:
    """Compute Add."""
    return record_op("Add", [x, y])


@register_op("ai.onnx", "Sub")
def sub(x: Tensor, y: Tensor) -> Tensor:
    """Compute Sub."""
    return record_op("Sub", [x, y])


@register_op("ai.onnx", "Mul")
def mul(x: Tensor, y: Tensor) -> Tensor:
    """Compute Mul."""
    return record_op("Mul", [x, y])


@register_op("ai.onnx", "Div")
def div(x: Tensor, y: Tensor) -> Tensor:
    """Compute Div."""
    return record_op("Div", [x, y])


@register_op("ai.onnx", "MatMul")
def matmul(x: Tensor, y: Tensor) -> Tensor:
    """Compute MatMul."""
    return record_op("MatMul", [x, y])


@register_op("ai.onnx", "Gemm")
def gemm(
    x: Tensor,
    y: Tensor,
    c: Tensor = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    trans_a: int = 0,
    trans_b: int = 0,
) -> Tensor:
    """Compute Gemm."""
    inputs = [x, y]
    if c is not None:
        inputs.append(c)
    attributes = {"alpha": alpha, "beta": beta, "trans_a": trans_a, "trans_b": trans_b}
    return record_op("Gemm", inputs, attributes)


@register_op("ai.onnx", "Einsum")
def einsum(inputs: list[Tensor], equation: str) -> Tensor:
    """Compute Einsum."""
    return record_op("Einsum", inputs, {"equation": equation})


@register_op("ai.onnx", "Erf")
def erf(input: Tensor) -> Tensor:
    """Compute Erf."""
    return record_op("Erf", [input])


@register_op("ai.onnx", "Exp")
def exp(input: Tensor) -> Tensor:
    """Compute Exp."""
    return record_op("Exp", [input])


@register_op("ai.onnx", "HammingWindow")
def hamming_window(x: Tensor) -> Tensor:
    """Compute HammingWindow."""
    return record_op("HammingWindow", [x])


@register_op("ai.onnx", "HannWindow")
def hann_window(x: Tensor) -> Tensor:
    """Compute HannWindow."""
    return record_op("HannWindow", [x])


@register_op("ai.onnx", "Log")
def log(x: Tensor) -> Tensor:
    """Compute Log."""
    return record_op("Log", [x])


@register_op("ai.onnx", "Sqrt")
def sqrt(x: Tensor) -> Tensor:
    """Compute Sqrt."""
    return record_op("Sqrt", [x])


@register_op("ai.onnx", "Xor")
def xor(x: Tensor) -> Tensor:
    """Compute Xor."""
    return record_op("Xor", [x])


@register_op("ai.onnx", "STFT")
def stft(
    signal: Tensor,
    frame_step: Tensor = None,
    window_length: Tensor = None,
    frame_length: Tensor = None,
    window: Tensor = None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Compute STFT."""
    inputs = [signal]
    if frame_step is not None:
        inputs.append(frame_step)
    if window_length is not None:
        inputs.append(window_length)
    if frame_length is not None:
        inputs.append(frame_length)
    if window is not None:
        inputs.append(window)
    return record_op("STFT", inputs, kwargs)


@register_op("ai.onnx", "ISTFT")
def istft(
    signal: Tensor,
    frame_step: Tensor = None,
    window_length: Tensor = None,
    frame_length: Tensor = None,
    window: Tensor = None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Compute ISTFT."""
    inputs = [signal]
    if frame_step is not None:
        inputs.append(frame_step)
    if window_length is not None:
        inputs.append(window_length)
    if frame_length is not None:
        inputs.append(frame_length)
    if window is not None:
        inputs.append(window)
    return record_op("ISTFT", inputs, kwargs)


@register_op("ai.onnx", "Fmod")
def fmod(x: Tensor, y: Tensor) -> Tensor:
    """Compute Fmod."""
    return record_op("Fmod", [x, y])


@register_op("ai.onnx", "Log2")
def log2(x: Tensor) -> Tensor:
    """Compute Log2."""
    return record_op("Log2", [x])


@register_op("ai.onnx", "Log10")
def log10(x: Tensor) -> Tensor:
    """Compute Log10."""
    return record_op("Log10", [x])


@register_op("ai.onnx", "Expm1")
def expm1(x: Tensor) -> Tensor:
    """Compute Expm1."""
    return record_op("Expm1", [x])


@register_op("ai.onnx", "Log1p")
def log1p(x: Tensor) -> Tensor:
    """Compute Log1p."""
    return record_op("Log1p", [x])


@register_op("ai.onnx", "IsFinite")
def isfinite(x: Tensor) -> Tensor:
    """Compute IsFinite."""
    return record_op("IsFinite", [x])


@register_op("ai.onnx", "LogicalAnd")
def logical_and(x: Tensor, y: Tensor) -> Tensor:
    """Compute LogicalAnd."""
    return record_op("LogicalAnd", [x, y])


@register_op("ai.onnx", "LogicalOr")
def logical_or(x: Tensor, y: Tensor) -> Tensor:
    """Compute LogicalOr."""
    return record_op("LogicalOr", [x, y])


@register_op("ai.onnx", "LogicalXor")
def logical_xor(x: Tensor, y: Tensor) -> Tensor:
    """Compute LogicalXor."""
    return record_op("LogicalXor", [x, y])


@register_op("ai.onnx", "LogicalNot")
def logical_not(x: Tensor) -> Tensor:
    """Compute LogicalNot."""
    return record_op("LogicalNot", [x])
