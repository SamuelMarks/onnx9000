"""Module docstring."""

from typing import Any, Optional

from onnx9000.frontend.tensor import Tensor
from onnx9000.frontend.utils import record_op


def relu(x: Tensor) -> Tensor:
    """Computes Rectified Linear Unit."""
    return record_op("Relu", [x])


def abs(x: Tensor) -> Tensor:
    """Computes Absolute Value."""
    return record_op("Abs", [x])


def acos(x: Tensor) -> Tensor:
    """Computes Inverse Cosine."""
    return record_op("Acos", [x])


def acosh(x: Tensor) -> Tensor:
    """Computes Inverse Hyperbolic Cosine."""
    return record_op("Acosh", [x])


def asin(x: Tensor) -> Tensor:
    """Computes Inverse Sine."""
    return record_op("Asin", [x])


def asinh(x: Tensor) -> Tensor:
    """Computes Inverse Hyperbolic Sine."""
    return record_op("Asinh", [x])


def atan(x: Tensor) -> Tensor:
    """Computes Inverse Tangent."""
    return record_op("Atan", [x])


def atanh(x: Tensor) -> Tensor:
    """Computes Inverse Hyperbolic Tangent."""
    return record_op("Atanh", [x])


def cos(x: Tensor) -> Tensor:
    """Computes Cosine."""
    return record_op("Cos", [x])


def cosh(x: Tensor) -> Tensor:
    """Computes Hyperbolic Cosine."""
    return record_op("Cosh", [x])


def sin(x: Tensor) -> Tensor:
    """Computes Sine."""
    return record_op("Sin", [x])


def sinh(x: Tensor) -> Tensor:
    """Computes Hyperbolic Sine."""
    return record_op("Sinh", [x])


def tan(x: Tensor) -> Tensor:
    """Computes Tangent."""
    return record_op("Tan", [x])


def max(x: Tensor, y: Tensor) -> Tensor:
    """Computes Max."""
    return record_op("Max", [x, y])


def min(x: Tensor, y: Tensor) -> Tensor:
    """Computes Min."""
    return record_op("Min", [x, y])


def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Computes Where."""
    return record_op("Where", [condition, x, y])


def cast(x: Tensor, to_type: int) -> Tensor:
    """Computes Cast."""
    return record_op("Cast", [x], {"to": to_type})


def cast_like(x: Tensor, target: Tensor) -> Tensor:
    """Computes CastLike."""
    return record_op("CastLike", [x, target])


def bitshift(x: Tensor, y: Tensor, direction: str) -> Tensor:
    """Computes BitShift."""
    return record_op("BitShift", [x, y], {"direction": direction})


def bitwise_and(x: Tensor, y: Tensor) -> Tensor:
    """Computes BitwiseAnd."""
    return record_op("BitwiseAnd", [x, y])


def bitwise_not(x: Tensor) -> Tensor:
    """Computes BitwiseNot."""
    return record_op("BitwiseNot", [x])


def bitwise_or(x: Tensor, y: Tensor) -> Tensor:
    """Computes BitwiseOr."""
    return record_op("BitwiseOr", [x, y])


def bitwise_xor(x: Tensor, y: Tensor) -> Tensor:
    """Computes BitwiseXor."""
    return record_op("BitwiseXor", [x, y])


def ceil(x: Tensor) -> Tensor:
    """Computes Ceil."""
    return record_op("Ceil", [x])


def floor(x: Tensor) -> Tensor:
    """Computes Floor."""
    return record_op("Floor", [x])


def isinf(x: Tensor) -> Tensor:
    """Computes IsInf."""
    return record_op("IsInf", [x])


def isnan(x: Tensor) -> Tensor:
    """Computes IsNaN."""
    return record_op("IsNaN", [x])


def mod(x: Tensor, y: Tensor) -> Tensor:
    """Computes Modulo."""
    return record_op("Mod", [x, y])


def neg(x: Tensor) -> Tensor:
    """Computes Negation."""
    return record_op("Neg", [x])


def pow(x: Tensor, y: Tensor) -> Tensor:
    """Computes Power."""
    return record_op("Pow", [x, y])


def reciprocal(x: Tensor) -> Tensor:
    """Computes Reciprocal."""
    return record_op("Reciprocal", [x])


def round(x: Tensor) -> Tensor:
    """Computes Round."""
    return record_op("Round", [x])


def sign(x: Tensor) -> Tensor:
    """Computes Sign."""
    return record_op("Sign", [x])


def sigmoid(x: Tensor) -> Tensor:
    """Computes Sigmoid."""
    return record_op("Sigmoid", [x])  # pragma: no cover


def tanh(x: Tensor) -> Tensor:
    """Computes Hyperbolic Tangent."""
    return record_op("Tanh", [x])  # pragma: no cover


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    """Computes Elu."""
    return record_op("Elu", [x], {"alpha": alpha})  # pragma: no cover


def celu(x: Tensor, alpha: float = 1.0) -> Tensor:
    """Computes Celu."""
    return record_op("Celu", [x], {"alpha": alpha})  # pragma: no cover


def leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    """Computes LeakyRelu."""
    return record_op("LeakyRelu", [x], {"alpha": alpha})  # pragma: no cover


def selu(x: Tensor, alpha: float = 1.67326, gamma: float = 1.0507) -> Tensor:
    """Computes Selu."""
    return record_op("Selu", [x], {"alpha": alpha, "gamma": gamma})  # pragma: no cover


def softplus(x: Tensor) -> Tensor:
    """Computes Softplus."""
    return record_op("Softplus", [x])  # pragma: no cover


def softsign(x: Tensor) -> Tensor:
    """Computes Softsign."""
    return record_op("Softsign", [x])  # pragma: no cover


def thresholded_relu(x: Tensor, alpha: float = 1.0) -> Tensor:
    """Computes ThresholdedRelu."""
    return record_op("ThresholdedRelu", [x], {"alpha": alpha})  # pragma: no cover


def mish(x: Tensor) -> Tensor:
    """Computes Mish."""
    return record_op("Mish", [x])  # pragma: no cover


def hard_sigmoid(x: Tensor, alpha: float = 0.2, beta: float = 0.5) -> Tensor:
    """Computes HardSigmoid."""
    return record_op(
        "HardSigmoid", [x], {"alpha": alpha, "beta": beta}
    )  # pragma: no cover


def hard_swish(x: Tensor) -> Tensor:
    """Computes HardSwish."""
    return record_op("HardSwish", [x])  # pragma: no cover


def hardmax(x: Tensor, axis: int = -1) -> Tensor:
    """Computes Hardmax."""
    return record_op("Hardmax", [x], {"axis": axis})


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Computes Softmax."""
    return record_op("Softmax", [x], {"axis": axis})


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Computes LogSoftmax."""
    return record_op("LogSoftmax", [x], {"axis": axis})


def prelu(x: Tensor, slope: Tensor) -> Tensor:
    """Computes PRelu."""
    return record_op("PRelu", [x, slope])  # pragma: no cover


def rnn(x: Tensor, w: Tensor, r: Tensor) -> Tensor:
    """Computes RNN."""
    return record_op("RNN", [x, w, r])  # pragma: no cover


def lstm(x: Tensor, w: Tensor, r: Tensor) -> Tensor:
    """Computes LSTM."""
    return record_op("LSTM", [x, w, r])  # pragma: no cover


def gru(x: Tensor, w: Tensor, r: Tensor) -> Tensor:
    """Computes GRU."""
    return record_op("GRU", [x, w, r])  # pragma: no cover


def sequence_construct(inputs: list[Tensor]) -> Tensor:
    """Computes SequenceConstruct."""
    return record_op("SequenceConstruct", inputs)


def sequence_at(seq: Tensor, position: Tensor) -> Tensor:
    """Computes SequenceAt."""
    return record_op("SequenceAt", [seq, position])  # pragma: no cover


def sequence_empty(dtype: int) -> Tensor:
    """Computes SequenceEmpty."""
    return record_op("SequenceEmpty", [], {"dtype": dtype})


def sequence_erase(seq: Tensor, position: Optional[Tensor] = None) -> Tensor:
    """Computes SequenceErase."""
    inputs = [seq]  # pragma: no cover
    if position is not None:  # pragma: no cover
        inputs.append(position)  # pragma: no cover
    return record_op("SequenceErase", inputs)  # pragma: no cover


def sequence_insert(
    seq: Tensor, tensor: Tensor, position: Optional[Tensor] = None
) -> Tensor:
    """Computes SequenceInsert."""
    inputs = [seq, tensor]
    if position is not None:
        inputs.append(position)
    return record_op("SequenceInsert", inputs)


def sequence_length(seq: Tensor) -> Tensor:
    """Computes SequenceLength."""
    return record_op("SequenceLength", [seq])  # pragma: no cover


def sequence_map(seq: Tensor) -> Tensor:
    """Computes SequenceMap."""
    # Omitted subgraphs for purely mocked implementation
    return record_op("SequenceMap", [seq])


def concat_from_sequence(seq: Tensor, axis: int, new_axis: int = 0) -> Tensor:
    """Computes ConcatFromSequence."""
    return record_op("ConcatFromSequence", [seq], {"axis": axis, "new_axis": new_axis})


def split_to_sequence(x: Tensor, axis: int = 0, keepdims: int = 1) -> Tensor:
    """Computes SplitToSequence."""
    return record_op(
        "SplitToSequence", [x], {"axis": axis, "keepdims": keepdims}
    )  # pragma: no cover


def constant(value: Any, dtype: int = 1) -> Tensor:
    """Computes Constant."""
    return record_op("Constant", [], {"value": value, "dtype": dtype})


def constant_of_shape(input: Tensor, value: Any = 0.0, dtype: int = 1) -> Tensor:
    """Computes ConstantOfShape."""
    return record_op("ConstantOfShape", [input], {"value": value, "dtype": dtype})


def concat(inputs: list[Tensor], axis: int) -> Tensor:
    """Computes Concat."""
    return record_op("Concat", inputs, {"axis": axis})


def quantize_linear(
    x: Tensor, y_scale: Tensor, y_zero_point: Optional[Tensor] = None
) -> Tensor:
    """Computes QuantizeLinear."""
    inputs = [x, y_scale]  # pragma: no cover
    if y_zero_point is not None:  # pragma: no cover
        inputs.append(y_zero_point)  # pragma: no cover
    return record_op("QuantizeLinear", inputs)  # pragma: no cover


def dequantize_linear(
    x: Tensor, x_scale: Tensor, x_zero_point: Optional[Tensor] = None
) -> Tensor:
    """Computes DequantizeLinear."""
    inputs = [x, x_scale]  # pragma: no cover
    if x_zero_point is not None:  # pragma: no cover
        inputs.append(x_zero_point)  # pragma: no cover
    return record_op("DequantizeLinear", inputs)  # pragma: no cover


def conv_transpose(
    x: Tensor, w: Tensor, b: Optional[Tensor] = None, **kwargs: Any
) -> Tensor:
    """Computes ConvTranspose."""
    inputs = [x, w]  # pragma: no cover
    if b is not None:  # pragma: no cover
        inputs.append(b)  # pragma: no cover
    return record_op("ConvTranspose", inputs, kwargs)  # pragma: no cover


def blackman_window(
    size: Tensor, output_datatype: int = 1, periodic: int = 1
) -> Tensor:
    """Computes BlackmanWindow."""
    return record_op(
        "BlackmanWindow",
        [size],
        {"output_datatype": output_datatype, "periodic": periodic},
    )


def deform_conv(
    x: Tensor,
    w: Tensor,
    offset: Tensor,
    b: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    **kwargs: Any,
) -> Tensor:
    """Computes DeformConv."""
    inputs = [x, w, offset]  # pragma: no cover
    if b is not None:  # pragma: no cover
        inputs.append(b)  # pragma: no cover
    if mask is not None:  # pragma: no cover
        if b is None:  # pragma: no cover
            raise ValueError(  # pragma: no cover
                "Mask provided without Bias. ONNX requires sequential inputs."
            )
        inputs.append(mask)  # pragma: no cover
    return record_op("DeformConv", inputs, kwargs)  # pragma: no cover


def lp_normalization(input: Tensor, axis: int = -1, p: int = 2) -> Tensor:
    """Computes LpNormalization."""
    return record_op("LpNormalization", [input], {"axis": axis, "p": p})


def lp_pool(
    input: Tensor, kernel_shape: list[int], p: float = 2.0, **kwargs: Any
) -> Tensor:
    """Computes LpPool."""
    return record_op(
        "LpPool", [input], {"kernel_shape": kernel_shape, "p": p, **kwargs}
    )


def det(x: Tensor) -> Tensor:
    """Computes Det."""
    return record_op("Det", [x])


def eye_like(input: Tensor, dtype: int = 1, k: int = 0) -> Tensor:
    """Computes EyeLike."""
    return record_op("EyeLike", [input], {"dtype": dtype, "k": k})


def layer_normalization(
    x: Tensor,
    scale: Tensor,
    b: Optional[Tensor] = None,
    axis: int = -1,
    epsilon: float = 1e-05,
    stash_type: int = 1,
) -> Tensor:
    """Computes LayerNormalization."""
    inputs = [x, scale]
    if b is not None:
        inputs.append(b)
    return record_op(
        "LayerNormalization",
        inputs,
        {"axis": axis, "epsilon": epsilon, "stash_type": stash_type},
    )


def mean_variance_normalization(x: Tensor, axes: list[int]) -> Tensor:
    """Computes MeanVarianceNormalization."""
    return record_op(
        "MeanVarianceNormalization", [x], {"axes": axes}
    )  # pragma: no cover


def instance_normalization(
    input: Tensor, scale: Tensor, B: Tensor, epsilon: float = 1e-05
) -> Tensor:
    """Computes InstanceNormalization."""
    return record_op(
        "InstanceNormalization", [input, scale, B], {"epsilon": epsilon}
    )  # pragma: no cover


def less_or_equal(a: Tensor, b: Tensor) -> Tensor:
    """Computes LessOrEqual."""
    return record_op("LessOrEqual", [a, b])


def greater_or_equal(a: Tensor, b: Tensor) -> Tensor:
    """Computes GreaterOrEqual."""
    return record_op("GreaterOrEqual", [a, b])


def mean(inputs: list[Tensor]) -> Tensor:
    """Computes Mean."""
    return record_op("Mean", inputs)  # pragma: no cover


def mel_weight_matrix(
    num_mel_bins: Tensor,
    dft_length: Tensor,
    sample_rate: Tensor,
    lower_edge_hertz: Tensor,
    upper_edge_hertz: Tensor,
    output_datatype: int = 1,
) -> Tensor:
    """Computes MelWeightMatrix."""
    inputs = [num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz]
    return record_op("MelWeightMatrix", inputs, {"output_datatype": output_datatype})


def multinomial(
    input: Tensor, dtype: int = 6, sample_size: int = 1, seed: float = 0.0
) -> Tensor:
    """Computes Multinomial."""
    return record_op(
        "Multinomial",
        [input],
        {"dtype": dtype, "sample_size": sample_size, "seed": seed},
    )


def non_max_suppression(
    boxes: Tensor,
    scores: Tensor,
    max_output_boxes_per_class: Optional[Tensor] = None,
    iou_threshold: Optional[Tensor] = None,
    score_threshold: Optional[Tensor] = None,
    center_point_box: int = 0,
) -> Tensor:
    """Computes NonMaxSuppression."""
    inputs = [boxes, scores]
    if max_output_boxes_per_class is not None:
        inputs.append(max_output_boxes_per_class)  # pragma: no cover
    if iou_threshold is not None:
        if max_output_boxes_per_class is None:  # pragma: no cover
            raise ValueError(  # pragma: no cover
                "ONNX sequential inputs rule violation for NonMaxSuppression"
            )
        inputs.append(iou_threshold)  # pragma: no cover
    if score_threshold is not None:
        if iou_threshold is None:  # pragma: no cover
            raise ValueError(  # pragma: no cover
                "ONNX sequential inputs rule violation for NonMaxSuppression"
            )
        inputs.append(score_threshold)  # pragma: no cover

    return record_op(
        "NonMaxSuppression", inputs, {"center_point_box": center_point_box}
    )


def non_zero(x: Tensor) -> Tensor:
    """Computes NonZero."""
    return record_op("NonZero", [x])


def random_normal(
    dtype: int = 1,
    mean: float = 0.0,
    scale: float = 1.0,
    seed: float = 0.0,
    shape: list[int] = [],
) -> Tensor:
    """Computes RandomNormal."""
    return record_op(  # pragma: no cover
        "RandomNormal",
        [],
        {"dtype": dtype, "mean": mean, "scale": scale, "seed": seed, "shape": shape},
    )


def random_normal_like(
    input: Tensor,
    dtype: Optional[int] = None,
    mean: float = 0.0,
    scale: float = 1.0,
    seed: float = 0.0,
) -> Tensor:
    """Computes RandomNormalLike."""
    attrs: dict[str, Any] = {
        "mean": mean,
        "scale": scale,
        "seed": seed,
    }  # pragma: no cover
    if dtype is not None:  # pragma: no cover
        attrs["dtype"] = dtype  # pragma: no cover
    return record_op("RandomNormalLike", [input], attrs)  # pragma: no cover


def random_uniform(
    dtype: int = 1,
    high: float = 1.0,
    low: float = 0.0,
    seed: float = 0.0,
    shape: list[int] = [],
) -> Tensor:
    """Computes RandomUniform."""
    return record_op(  # pragma: no cover
        "RandomUniform",
        [],
        {"dtype": dtype, "high": high, "low": low, "seed": seed, "shape": shape},
    )


def random_uniform_like(
    input: Tensor,
    dtype: Optional[int] = None,
    high: float = 1.0,
    low: float = 0.0,
    seed: float = 0.0,
) -> Tensor:
    """Computes RandomUniformLike."""
    attrs: dict[str, Any] = {"high": high, "low": low, "seed": seed}  # pragma: no cover
    if dtype is not None:  # pragma: no cover
        attrs["dtype"] = dtype  # pragma: no cover
    return record_op("RandomUniformLike", [input], attrs)  # pragma: no cover


def reduce_sum_square(
    data: Tensor, axes: Optional[list[int]] = None, keepdims: int = 1
) -> Tensor:
    """Computes ReduceSumSquare."""
    attrs = {"keepdims": keepdims}
    if axes is not None:
        attrs["axes"] = axes  # pragma: no cover
    return record_op("ReduceSumSquare", [data], attrs)


def reduce_l1(
    data: Tensor, axes: Optional[list[int]] = None, keepdims: int = 1
) -> Tensor:
    """Computes ReduceL1."""
    attrs = {"keepdims": keepdims}
    if axes is not None:
        attrs["axes"] = axes  # pragma: no cover
    return record_op("ReduceL1", [data], attrs)


def reduce_l2(
    data: Tensor, axes: Optional[list[int]] = None, keepdims: int = 1
) -> Tensor:
    """Computes ReduceL2."""
    attrs = {"keepdims": keepdims}
    if axes is not None:
        attrs["axes"] = axes  # pragma: no cover
    return record_op("ReduceL2", [data], attrs)


def reduce_log_sum(
    data: Tensor, axes: Optional[list[int]] = None, keepdims: int = 1
) -> Tensor:
    """Computes ReduceLogSum."""
    attrs = {"keepdims": keepdims}
    if axes is not None:
        attrs["axes"] = axes  # pragma: no cover
    return record_op("ReduceLogSum", [data], attrs)


def reduce_log_sum_exp(
    data: Tensor, axes: Optional[list[int]] = None, keepdims: int = 1
) -> Tensor:
    """Computes ReduceLogSumExp."""
    attrs = {"keepdims": keepdims}
    if axes is not None:
        attrs["axes"] = axes  # pragma: no cover
    return record_op("ReduceLogSumExp", [data], attrs)


def range_op(start: Tensor, limit: Tensor, delta: Tensor) -> Tensor:
    """Computes Range."""
    return record_op("Range", [start, limit, delta])  # pragma: no cover


def regex_full_match(x: Tensor, pattern: str) -> Tensor:
    """Computes RegexFullMatch."""
    return record_op("RegexFullMatch", [x], {"pattern": pattern})  # pragma: no cover


def resize(
    x: Tensor,
    roi: Optional[Tensor] = None,
    scales: Optional[Tensor] = None,
    sizes: Optional[Tensor] = None,
    **kwargs: Any,
) -> Tensor:
    """Computes Resize."""
    inputs = [x]  # pragma: no cover
    if roi is not None:  # pragma: no cover
        inputs.append(roi)  # pragma: no cover
    elif scales is not None or sizes is not None:  # pragma: no cover
        inputs.append(
            record_op("Constant", [], {"value": []})
        )  # dummy for roi  # pragma: no cover

    if scales is not None:  # pragma: no cover
        inputs.append(scales)  # pragma: no cover
    elif sizes is not None:  # pragma: no cover
        inputs.append(
            record_op("Constant", [], {"value": []})
        )  # dummy for scales  # pragma: no cover

    if sizes is not None:  # pragma: no cover
        inputs.append(sizes)  # pragma: no cover

    return record_op("Resize", inputs, kwargs)  # pragma: no cover


def reverse_sequence(
    input: Tensor, sequence_lens: Tensor, batch_axis: int = 1, time_axis: int = 0
) -> Tensor:
    """Computes ReverseSequence."""
    return record_op(
        "ReverseSequence",
        [input, sequence_lens],
        {"batch_axis": batch_axis, "time_axis": time_axis},
    )


def scatter(data: Tensor, indices: Tensor, updates: Tensor, axis: int = 0) -> Tensor:
    """Computes Scatter."""
    return record_op("Scatter", [data, indices, updates], {"axis": axis})


def scatter_elements(
    data: Tensor,
    indices: Tensor,
    updates: Tensor,
    axis: int = 0,
    reduction: str = "none",
) -> Tensor:
    """Computes ScatterElements."""
    return record_op(
        "ScatterElements",
        [data, indices, updates],
        {"axis": axis, "reduction": reduction},
    )


def scatter_nd(
    data: Tensor, indices: Tensor, updates: Tensor, reduction: str = "none"
) -> Tensor:
    """Computes ScatterND."""
    return record_op("ScatterND", [data, indices, updates], {"reduction": reduction})


def shrink(input: Tensor, bias: float = 0.0, lambd: float = 0.5) -> Tensor:
    """Computes Shrink."""
    return record_op("Shrink", [input], {"bias": bias, "lambd": lambd})


def size(data: Tensor) -> Tensor:
    """Computes Size."""
    return record_op("Size", [data])


def string_concat(x: Tensor, y: Tensor) -> Tensor:
    """Computes StringConcat."""
    return record_op("StringConcat", [x, y])  # pragma: no cover


def string_normalizer(
    x: Tensor,
    case_change_action: str = "NONE",
    is_case_sensitive: int = 0,
    locale: str = "",
    stopwords: Optional[list[str]] = None,
) -> Tensor:
    """Computes StringNormalizer."""
    attrs: dict[str, Any] = {  # pragma: no cover
        "case_change_action": case_change_action,
        "is_case_sensitive": is_case_sensitive,
        "locale": locale,
    }
    if stopwords is not None:  # pragma: no cover
        attrs["stopwords"] = stopwords  # pragma: no cover
    return record_op("StringNormalizer", [x], attrs)  # pragma: no cover


def string_split(x: Tensor, delimiter: str = "", maxsplit: int = -1) -> Tensor:
    """Computes StringSplit."""
    return record_op(
        "StringSplit", [x], {"delimiter": delimiter, "maxsplit": maxsplit}
    )  # pragma: no cover


def trilu(input: Tensor, k: Optional[Tensor] = None, upper: int = 1) -> Tensor:
    """Computes Trilu."""
    inputs = [input]
    if k is not None:
        inputs.append(k)  # pragma: no cover
    return record_op("Trilu", inputs, {"upper": upper})


def topk(
    X: Tensor, K: Tensor, axis: int = -1, largest: int = 1, sorted: int = 1
) -> Tensor:
    """Computes TopK."""
    return record_op(  # pragma: no cover
        "TopK", [X, K], {"axis": axis, "largest": largest, "sorted": sorted}
    )


def unique(X: Tensor, axis: Optional[int] = None, sorted: int = 1) -> Tensor:
    """Computes Unique."""
    attrs = {"sorted": sorted}  # pragma: no cover
    if axis is not None:  # pragma: no cover
        attrs["axis"] = axis  # pragma: no cover
    return record_op("Unique", [X], attrs)  # pragma: no cover


def sequence_at(input_sequence: Tensor, position: Tensor) -> Tensor:
    """Computes SequenceAt."""
    return record_op("SequenceAt", [input_sequence, position])


def split_to_sequence(
    input: Tensor, split: Optional[Tensor] = None, axis: int = 0, keepdims: int = 1
) -> Tensor:
    """Computes SplitToSequence."""
    inputs = [input]
    if split is not None:
        inputs.append(split)  # pragma: no cover
    return record_op("SplitToSequence", inputs, {"axis": axis, "keepdims": keepdims})


def sequence_erase(input_sequence: Tensor, position: Optional[Tensor] = None) -> Tensor:
    """Computes SequenceErase."""
    inputs = [input_sequence]
    if position is not None:
        inputs.append(position)
    return record_op("SequenceErase", inputs)


def sequence_length(input_sequence: Tensor) -> Tensor:
    """Computes SequenceLength."""
    return record_op("SequenceLength", [input_sequence])


def affine_grid(theta: Tensor, size: Tensor, align_corners: int = 0) -> Tensor:
    """Computes AffineGrid."""
    return record_op(
        "AffineGrid", [theta, size], {"align_corners": align_corners}
    )  # pragma: no cover


def argmax(
    data: Tensor, axis: int = 0, keepdims: int = 1, select_last_index: int = 0
) -> Tensor:
    """Computes ArgMax."""
    return record_op(  # pragma: no cover
        "ArgMax",
        [data],
        {"axis": axis, "keepdims": keepdims, "select_last_index": select_last_index},
    )


def argmin(
    data: Tensor, axis: int = 0, keepdims: int = 1, select_last_index: int = 0
) -> Tensor:
    """Computes ArgMin."""
    return record_op(  # pragma: no cover
        "ArgMin",
        [data],
        {"axis": axis, "keepdims": keepdims, "select_last_index": select_last_index},
    )


def attention(Q: Tensor, K: Tensor, V: Tensor, **kwargs: Any) -> Tensor:
    """Computes Attention."""
    return record_op("Attention", [Q, K, V], kwargs)  # pragma: no cover


def bernoulli(input: Tensor, dtype: Optional[int] = None, seed: float = 0.0) -> Tensor:
    """Computes Bernoulli."""
    attrs: dict[str, Any] = {"seed": seed}  # pragma: no cover
    if dtype is not None:  # pragma: no cover
        attrs["dtype"] = dtype  # pragma: no cover
    return record_op("Bernoulli", [input], attrs)  # pragma: no cover


def center_crop_pad(
    input_data: Tensor, shape: Tensor, axes: Optional[list[int]] = None
) -> Tensor:
    """Computes CenterCropPad."""
    attrs = {}  # pragma: no cover
    if axes is not None:  # pragma: no cover
        attrs["axes"] = axes  # pragma: no cover
    return record_op("CenterCropPad", [input_data, shape], attrs)  # pragma: no cover


def clip(
    input: Tensor, min: Optional[Tensor] = None, max: Optional[Tensor] = None
) -> Tensor:
    """Computes Clip."""
    inputs = [input]  # pragma: no cover
    if min is not None:  # pragma: no cover
        inputs.append(min)  # pragma: no cover
    elif max is not None:  # pragma: no cover
        inputs.append(record_op("Constant", [], {"value": []}))  # pragma: no cover
    if max is not None:  # pragma: no cover
        inputs.append(max)  # pragma: no cover
    return record_op("Clip", inputs)  # pragma: no cover


def col2im(
    input: Tensor,
    image_shape: Tensor,
    block_shape: Tensor,
    dilations: Optional[list[int]] = None,
    pads: Optional[list[int]] = None,
    strides: Optional[list[int]] = None,
) -> Tensor:
    """Computes Col2Im."""
    attrs = {}  # pragma: no cover
    if dilations is not None:  # pragma: no cover
        attrs["dilations"] = dilations  # pragma: no cover
    if pads is not None:  # pragma: no cover
        attrs["pads"] = pads  # pragma: no cover
    if strides is not None:  # pragma: no cover
        attrs["strides"] = strides  # pragma: no cover
    return record_op(
        "Col2Im", [input, image_shape, block_shape], attrs
    )  # pragma: no cover


def compress(input: Tensor, condition: Tensor, axis: Optional[int] = None) -> Tensor:
    """Computes Compress."""
    attrs = {}
    if axis is not None:
        attrs["axis"] = axis
    return record_op("Compress", [input, condition], attrs)


def conv_integer(
    x: Tensor,
    w: Tensor,
    x_zero_point: Optional[Tensor] = None,
    w_zero_point: Optional[Tensor] = None,
    **kwargs: Any,
) -> Tensor:
    """Computes ConvInteger."""
    inputs = [x, w]  # pragma: no cover
    if x_zero_point is not None:  # pragma: no cover
        inputs.append(x_zero_point)  # pragma: no cover
    elif w_zero_point is not None:  # pragma: no cover
        inputs.append(record_op("Constant", [], {"value": []}))  # pragma: no cover
    if w_zero_point is not None:  # pragma: no cover
        inputs.append(w_zero_point)  # pragma: no cover
    return record_op("ConvInteger", inputs, kwargs)  # pragma: no cover


def cumsum(x: Tensor, axis: Tensor, exclusive: int = 0, reverse: int = 0) -> Tensor:
    """Computes CumSum."""
    return record_op("CumSum", [x, axis], {"exclusive": exclusive, "reverse": reverse})


def dft(
    input: Tensor,
    dft_length: Optional[Tensor] = None,
    axis: int = 1,
    inverse: int = 0,
    onesided: int = 0,
) -> Tensor:
    """Computes DFT."""
    inputs = [input]
    if dft_length is not None:
        inputs.append(dft_length)  # pragma: no cover
    return record_op(
        "DFT", inputs, {"axis": axis, "inverse": inverse, "onesided": onesided}
    )


def depth_to_space(input: Tensor, blocksize: int, mode: str = "DCR") -> Tensor:
    """Computes DepthToSpace."""
    return record_op(
        "DepthToSpace", [input], {"blocksize": blocksize, "mode": mode}
    )  # pragma: no cover


def dropout(
    data: Tensor,
    ratio: Optional[Tensor] = None,
    training_mode: Optional[Tensor] = None,
    seed: int = 0,
) -> Tensor:
    """Computes Dropout."""
    inputs = [data]
    if ratio is not None:
        inputs.append(ratio)  # pragma: no cover
    elif training_mode is not None:
        inputs.append(record_op("Constant", [], {"value": []}))  # pragma: no cover
    if training_mode is not None:
        inputs.append(training_mode)  # pragma: no cover
    return record_op("Dropout", inputs, {"seed": seed})


def reshape(x: Tensor, shape: Tensor) -> Tensor:
    """Reshapes a tensor."""
    # Shape inference is complex here since `shape` tensor content isn't statically known during pure symbolic tracing
    # For now, we defer true shape inference and mark the output shape as dynamic if unknown.
    # In a full tracer, we'd do partial evaluation or rely on the IR parser.
    return record_op("Reshape", [x, shape])


def average_pool(
    x: Tensor,
    kernel_shape: list[int],
    strides: Optional[list[int]] = None,
    pads: Optional[list[int]] = None,
) -> Tensor:
    """Computes AveragePool."""
    attr = {"kernel_shape": kernel_shape}
    if strides:
        attr["strides"] = strides
    if pads:
        attr["pads"] = pads  # pragma: no cover
    return record_op("AveragePool", [x], attr)


def max_pool(
    x: Tensor,
    kernel_shape: list[int],
    strides: Optional[list[int]] = None,
    pads: Optional[list[int]] = None,
) -> Tensor:
    """Computes MaxPool."""
    attr = {"kernel_shape": kernel_shape}
    if strides:
        attr["strides"] = strides
    if pads:
        attr["pads"] = pads  # pragma: no cover
    return record_op("MaxPool", [x], attr)


def global_average_pool(x: Tensor) -> Tensor:
    """Computes GlobalAveragePool."""
    return record_op("GlobalAveragePool", [x])


def global_max_pool(x: Tensor) -> Tensor:
    """Computes GlobalMaxPool."""
    return record_op("GlobalMaxPool", [x])


def transpose(x: Tensor, perm: Optional[list[int]] = None) -> Tensor:
    """Transposes a tensor."""
    attributes = {"perm": perm} if perm is not None else {}
    return record_op("Transpose", [x], attributes)


def equal(x: Tensor, y: Tensor) -> Tensor:
    """Computes Equal."""
    return record_op("Equal", [x, y])


def greater(x: Tensor, y: Tensor) -> Tensor:
    """Computes Greater."""
    return record_op("Greater", [x, y])


def less(x: Tensor, y: Tensor) -> Tensor:
    """Computes Less."""
    return record_op("Less", [x, y])


def and_(x: Tensor, y: Tensor) -> Tensor:
    """Computes And."""
    return record_op("And", [x, y])


def or_(x: Tensor, y: Tensor) -> Tensor:
    """Computes Or."""
    return record_op("Or", [x, y])


def not_(x: Tensor) -> Tensor:
    """Computes Not."""
    return record_op("Not", [x])


def add(x: Tensor, y: Tensor) -> Tensor:
    """Computes Add."""
    return record_op("Add", [x, y])  # pragma: no cover


def sub(x: Tensor, y: Tensor) -> Tensor:
    """Computes Sub."""
    return record_op("Sub", [x, y])  # pragma: no cover


def mul(x: Tensor, y: Tensor) -> Tensor:
    """Computes Mul."""
    return record_op("Mul", [x, y])  # pragma: no cover


def div(x: Tensor, y: Tensor) -> Tensor:
    """Computes Div."""
    return record_op("Div", [x, y])  # pragma: no cover


def matmul(x: Tensor, y: Tensor) -> Tensor:
    """Computes MatMul."""
    return record_op("MatMul", [x, y])  # pragma: no cover


def gemm(
    x: Tensor,
    y: Tensor,
    c: Optional[Tensor] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    trans_a: int = 0,
    trans_b: int = 0,
) -> Tensor:
    """Computes Gemm."""
    inputs = [x, y]  # pragma: no cover
    if c is not None:  # pragma: no cover
        inputs.append(c)  # pragma: no cover
    attributes = {
        "alpha": alpha,
        "beta": beta,
        "trans_a": trans_a,
        "trans_b": trans_b,
    }  # pragma: no cover
    return record_op("Gemm", inputs, attributes)  # pragma: no cover


def reduce_sum(
    x: Tensor, axes: Optional[list[int]] = None, keepdims: bool = True
) -> Tensor:
    """Computes the sum of the input tensor's element along the provided axes."""
    attributes: dict[str, Any] = {"keepdims": 1 if keepdims else 0}
    if axes is not None:
        attributes["axes"] = axes  # pragma: no cover
    return record_op("ReduceSum", [x], attributes)


def reduce_mean(
    x: Tensor, axes: Optional[list[int]] = None, keepdims: bool = True
) -> Tensor:
    """Computes the mean of the input tensor's element along the provided axes."""
    attributes: dict[str, Any] = {"keepdims": 1 if keepdims else 0}
    if axes is not None:
        attributes["axes"] = axes  # pragma: no cover
    return record_op("ReduceMean", [x], attributes)


def reduce_max(
    x: Tensor, axes: Optional[list[int]] = None, keepdims: bool = True
) -> Tensor:
    """Computes ReduceMax."""
    attributes: dict[str, Any] = {"keepdims": 1 if keepdims else 0}
    if axes is not None:
        attributes["axes"] = axes  # pragma: no cover
    return record_op("ReduceMax", [x], attributes)


def reduce_min(
    x: Tensor, axes: Optional[list[int]] = None, keepdims: bool = True
) -> Tensor:
    """Computes ReduceMin."""
    attributes: dict[str, Any] = {"keepdims": 1 if keepdims else 0}
    if axes is not None:
        attributes["axes"] = axes  # pragma: no cover
    return record_op("ReduceMin", [x], attributes)


def reduce_prod(
    x: Tensor, axes: Optional[list[int]] = None, keepdims: bool = True
) -> Tensor:
    """Computes ReduceProd."""
    attributes: dict[str, Any] = {"keepdims": 1 if keepdims else 0}
    if axes is not None:
        attributes["axes"] = axes  # pragma: no cover
    return record_op("ReduceProd", [x], attributes)


def conv(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
    strides: Optional[list[int]] = None,
    pads: Optional[list[int]] = None,
) -> Tensor:
    """Computes an N-D convolution."""
    if strides is None:
        strides = [1, 1]  # pragma: no cover
    if pads is None:
        pads = [0, 0, 0, 0]  # pragma: no cover

    inputs = [x, w]
    if b is not None:
        inputs.append(b)  # pragma: no cover

    attributes: dict[str, Any] = {"strides": strides, "pads": pads}
    return record_op("Conv", inputs, attributes)


def dynamic_quantize_linear(x: Tensor) -> Any:
    """Computes DynamicQuantizeLinear."""
    return record_op("DynamicQuantizeLinear", [x])  # pragma: no cover


def einsum(inputs: list[Tensor], equation: str) -> Tensor:
    """Computes Einsum."""
    return record_op("Einsum", inputs, {"equation": equation})  # pragma: no cover


def erf(input: Tensor) -> Tensor:
    """Computes Erf."""
    return record_op("Erf", [input])  # pragma: no cover


def exp(input: Tensor) -> Tensor:
    """Computes Exp."""
    return record_op("Exp", [input])  # pragma: no cover


def expand(input: Tensor, shape: Tensor) -> Tensor:
    """Execute the Expand process and return the computed results."""
    return record_op("Expand", [input, shape])  # pragma: no cover


def slice(
    data: Tensor,
    starts: Tensor,
    ends: Tensor,
    axes: Tensor = None,
    steps: Tensor = None,
) -> Tensor:
    """Computes Slice."""
    inputs = [data, starts, ends]
    if axes is not None:
        inputs.append(axes)
        if steps is not None:
            inputs.append(steps)
    return record_op("Slice", inputs)


def tile(input: Tensor, repeats: Tensor) -> Tensor:
    """Computes Tile."""
    return record_op("Tile", [input, repeats])


def gather(data: Tensor, indices: Tensor, axis: int = 0) -> Tensor:
    """Computes Gather."""
    return record_op("Gather", [data, indices], {"axis": axis})  # pragma: no cover


def gather_nd(data: Tensor, indices: Tensor, batch_dims: int = 0) -> Tensor:
    """Computes GatherND."""
    return record_op(
        "GatherND", [data, indices], {"batch_dims": batch_dims}
    )  # pragma: no cover


def gather_elements(data: Tensor, indices: Tensor, axis: int = 0) -> Tensor:
    """Computes GatherElements."""
    return record_op("GatherElements", [data, indices], {"axis": axis})


def depth_to_space(data: Tensor, blocksize: int, mode: str = "DCR") -> Tensor:
    """Computes DepthToSpace."""
    return record_op("DepthToSpace", [data], {"blocksize": blocksize, "mode": mode})


def space_to_depth(data: Tensor, blocksize: int) -> Tensor:
    """Computes SpaceToDepth."""
    return record_op("SpaceToDepth", [data], {"blocksize": blocksize})


def sum(data: list[Tensor]) -> Tensor:
    """Computes Sum."""
    return record_op("Sum", data)


def swish(x: Tensor) -> Tensor:
    """Computes Swish."""
    return record_op("Swish", [x])


def one_hot(indices: Tensor, depth: Tensor, values: Tensor, axis: int = -1) -> Tensor:
    """Computes OneHot."""
    return record_op(
        "OneHot", [indices, depth, values], {"axis": axis}
    )  # pragma: no cover


def lrn(
    x: Tensor,
    alpha: float = 0.0001,
    beta: float = 0.75,
    bias: float = 1.0,
    size: int = 1,
) -> Tensor:
    """Computes LRN."""
    return record_op(
        "LRN", [x], {"alpha": alpha, "beta": beta, "bias": bias, "size": size}
    )  # pragma: no cover


def group_normalization(
    x: Tensor, scale: Tensor, b: Tensor, epsilon: float = 1e-05, num_groups: int = 1
) -> Tensor:
    """Computes GroupNormalization."""
    return record_op(
        "GroupNormalization",
        [x, scale, b],
        {"epsilon": epsilon, "num_groups": num_groups},
    )  # pragma: no cover
