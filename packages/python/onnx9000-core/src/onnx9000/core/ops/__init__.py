"""Module providing core logic and structural definitions."""

from typing import Any, Optional

from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.registry import register_op


def record_op(
    op_type: str, inputs: list[Any], attributes: Optional[dict[str, Any]] = None
) -> Tensor:
    """Record the operation and returns a dummy output tensor."""
    attributes = attributes or {}
    tensors = [inp for inp in inputs if hasattr(inp, "dtype")]
    dtype = tensors[0].dtype if tensors else None
    return Tensor(name=f"{op_type}_out", shape=(), dtype=dtype)


from onnx9000.core.ops.torch_auto import *


@register_op("ai.onnx", "Relu")
def relu(x: Tensor) -> Tensor:
    """Compute Rectified Linear Unit."""
    return record_op("Relu", [x])


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


@register_op("ai.onnx", "Where")
def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Compute Where."""
    return record_op("Where", [condition, x, y])


@register_op("ai.onnx", "Cast")
def cast(x: Tensor, to_type: int) -> Tensor:
    """Compute Cast."""
    return record_op("Cast", [x], {"to": to_type})


@register_op("ai.onnx", "CastLike")
def cast_like(x: Tensor, target: Tensor) -> Tensor:
    """Compute CastLike."""
    return record_op("CastLike", [x, target])


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


@register_op("ai.onnx", "Sigmoid")
def sigmoid(x: Tensor) -> Tensor:
    """Compute Sigmoid."""
    return record_op("Sigmoid", [x])


@register_op("ai.onnx", "Tanh")
def tanh(x: Tensor) -> Tensor:
    """Compute Hyperbolic Tangent."""
    return record_op("Tanh", [x])


@register_op("ai.onnx", "Elu")
def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    """Compute Elu."""
    return record_op("Elu", [x], {"alpha": alpha})


@register_op("ai.onnx", "Celu")
def celu(x: Tensor, alpha: float = 1.0) -> Tensor:
    """Compute Celu."""
    return record_op("Celu", [x], {"alpha": alpha})


@register_op("ai.onnx", "LeakyRelu")
def leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    """Compute LeakyRelu."""
    return record_op("LeakyRelu", [x], {"alpha": alpha})


@register_op("ai.onnx", "Selu")
def selu(x: Tensor, alpha: float = 1.67326, gamma: float = 1.0507) -> Tensor:
    """Compute Selu."""
    return record_op("Selu", [x], {"alpha": alpha, "gamma": gamma})


@register_op("ai.onnx", "Softplus")
def softplus(x: Tensor) -> Tensor:
    """Compute Softplus."""
    return record_op("Softplus", [x])


@register_op("ai.onnx", "Softsign")
def softsign(x: Tensor) -> Tensor:
    """Compute Softsign."""
    return record_op("Softsign", [x])


@register_op("ai.onnx", "ThresholdedRelu")
def thresholded_relu(x: Tensor, alpha: float = 1.0) -> Tensor:
    """Compute ThresholdedRelu."""
    return record_op("ThresholdedRelu", [x], {"alpha": alpha})


@register_op("ai.onnx", "Mish")
def mish(x: Tensor) -> Tensor:
    """Compute Mish."""
    return record_op("Mish", [x])


@register_op("ai.onnx", "HardSigmoid")
def hard_sigmoid(x: Tensor, alpha: float = 0.2, beta: float = 0.5) -> Tensor:
    """Compute HardSigmoid."""
    return record_op("HardSigmoid", [x], {"alpha": alpha, "beta": beta})


@register_op("ai.onnx", "HardSwish")
def hard_swish(x: Tensor) -> Tensor:
    """Compute HardSwish."""
    return record_op("HardSwish", [x])


@register_op("ai.onnx", "Hardmax")
def hardmax(x: Tensor, axis: int = -1) -> Tensor:
    """Compute Hardmax."""
    return record_op("Hardmax", [x], {"axis": axis})


@register_op("ai.onnx", "Softmax")
def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Compute Softmax."""
    return record_op("Softmax", [x], {"axis": axis})


@register_op("ai.onnx", "LogSoftmax")
def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Compute LogSoftmax."""
    return record_op("LogSoftmax", [x], {"axis": axis})


@register_op("ai.onnx", "PRelu")
def prelu(x: Tensor, slope: Tensor) -> Tensor:
    """Compute PRelu."""
    return record_op("PRelu", [x, slope])


@register_op("ai.onnx", "RNN")
def rnn(x: Tensor, w: Tensor, r: Tensor) -> Tensor:
    """Compute RNN."""
    return record_op("RNN", [x, w, r])


@register_op("ai.onnx", "LSTM")
def lstm(x: Tensor, w: Tensor, r: Tensor) -> Tensor:
    """Compute LSTM."""
    return record_op("LSTM", [x, w, r])


@register_op("ai.onnx", "GRU")
def gru(x: Tensor, w: Tensor, r: Tensor) -> Tensor:
    """Compute GRU."""
    return record_op("GRU", [x, w, r])


@register_op("ai.onnx", "SequenceConstruct")
def sequence_construct(inputs: list[Tensor]) -> Tensor:
    """Compute SequenceConstruct."""
    return record_op("SequenceConstruct", inputs)


@register_op("ai.onnx", "SequenceEmpty")
def sequence_empty(dtype: int) -> Tensor:
    """Compute SequenceEmpty."""
    return record_op("SequenceEmpty", [], {"dtype": dtype})


@register_op("ai.onnx", "SequenceInsert")
def sequence_insert(seq: Tensor, tensor: Tensor, position: Optional[Tensor] = None) -> Tensor:
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


@register_op("ai.onnx", "Constant")
def constant(value: Any, dtype: int = 1) -> Tensor:
    """Compute Constant."""
    return record_op("Constant", [], {"value": value, "dtype": dtype})


@register_op("ai.onnx", "ConstantOfShape")
def constant_of_shape(input: Tensor, value: Any = 0.0, dtype: int = 1) -> Tensor:
    """Compute ConstantOfShape."""
    return record_op("ConstantOfShape", [input], {"value": value, "dtype": dtype})


@register_op("ai.onnx", "Concat")
def concat(inputs: list[Tensor], axis: int) -> Tensor:
    """Compute Concat."""
    return record_op("Concat", inputs, {"axis": axis})


@register_op("ai.onnx", "ConvTranspose")
def conv_transpose(x: Tensor, w: Tensor, b: Optional[Tensor] = None, **kwargs: Any) -> Tensor:
    """Compute ConvTranspose."""
    inputs = [x, w]
    if b is not None:
        inputs.append(b)
    return record_op("ConvTranspose", inputs, kwargs)


@register_op("ai.onnx", "BlackmanWindow")
def blackman_window(size: Tensor, output_datatype: int = 1, periodic: int = 1) -> Tensor:
    """Compute BlackmanWindow."""
    return record_op(
        "BlackmanWindow",
        [size],
        {"output_datatype": output_datatype, "periodic": periodic},
    )


@register_op("ai.onnx", "LpNormalization")
def lp_normalization(input: Tensor, axis: int = -1, p: int = 2) -> Tensor:
    """Compute LpNormalization."""
    return record_op("LpNormalization", [input], {"axis": axis, "p": p})


@register_op("ai.onnx", "LpPool")
def lp_pool(input: Tensor, kernel_shape: list[int], p: float = 2.0, **kwargs: Any) -> Tensor:
    """Compute LpPool."""
    return record_op("LpPool", [input], {"kernel_shape": kernel_shape, "p": p, **kwargs})


@register_op("ai.onnx", "Det")
def det(x: Tensor) -> Tensor:
    """Compute Det."""
    return record_op("Det", [x])


@register_op("ai.onnx", "EyeLike")
def eye_like(input: Tensor, dtype: int = 1, k: int = 0) -> Tensor:
    """Compute EyeLike."""
    return record_op("EyeLike", [input], {"dtype": dtype, "k": k})


@register_op("ai.onnx", "LayerNormalization")
def layer_normalization(
    x: Tensor,
    scale: Tensor,
    b: Optional[Tensor] = None,
    axis: int = -1,
    epsilon: float = 1e-05,
    stash_type: int = 1,
) -> Tensor:
    """Compute LayerNormalization."""
    inputs = [x, scale]
    if b is not None:
        inputs.append(b)
    return record_op(
        "LayerNormalization",
        inputs,
        {"axis": axis, "epsilon": epsilon, "stash_type": stash_type},
    )


@register_op("ai.onnx", "MeanVarianceNormalization")
def mean_variance_normalization(x: Tensor, axes: list[int]) -> Tensor:
    """Compute MeanVarianceNormalization."""
    return record_op("MeanVarianceNormalization", [x], {"axes": axes})


@register_op("ai.onnx", "InstanceNormalization")
def instance_normalization(
    input: Tensor, scale: Tensor, B: Tensor, epsilon: float = 1e-05
) -> Tensor:
    """Compute InstanceNormalization."""
    return record_op("InstanceNormalization", [input, scale, B], {"epsilon": epsilon})


@register_op("onnx9000.custom", "KVCache")
def kv_cache(
    cache: Tensor, new_key: Tensor, new_value: Tensor, **kwargs: Any
) -> tuple[Tensor, Tensor]:
    """Compute Key-Value Cache update."""
    return record_op("KVCache", [cache, new_key, new_value], kwargs)


@register_op("onnx9000.custom", "RoPE")
def rope(x: Tensor, cos: Tensor, sin: Tensor, **kwargs: Any) -> Tensor:
    """Compute Rotary Positional Embeddings."""
    return record_op("RoPE", [x, cos, sin], kwargs)


@register_op("onnx9000.custom", "DeformConv")
def deform_conv_custom(
    x: Tensor, offset: Tensor, w: Tensor, b: Optional[Tensor] = None, **kwargs: Any
) -> Tensor:
    """Compute Deformable Convolution."""
    inputs = [x, offset, w]
    if b is not None:
        inputs.append(b)
    return record_op("DeformConv", inputs, kwargs)


@register_op("onnx9000.custom", "DCT")
def dct(x: Tensor, **kwargs: Any) -> Tensor:
    """Compute Discrete Cosine Transform."""
    return record_op("DCT", [x], kwargs)


@register_op("onnx9000.custom", "Roll")
def roll(x: Tensor, **kwargs: Any) -> Tensor:
    """Compute Roll for windowed attention."""
    return record_op("Roll", [x], kwargs)


@register_op("onnx9000.custom", "Unroll")
def unroll(x: Tensor, **kwargs: Any) -> Tensor:
    """Compute Unroll for windowed attention."""
    return record_op("Unroll", [x], kwargs)


@register_op("onnx9000.custom", "Tokenizer")
def tokenizer(x: Tensor, **kwargs: Any) -> Tensor:
    """Compute Tokenization."""
    return record_op("Tokenizer", [x], kwargs)


@register_op("onnx9000.custom", "TextVectorization")
def text_vectorization(x: Tensor, **kwargs: Any) -> Tensor:
    """Compute Text Vectorization."""
    return record_op("TextVectorization", [x], kwargs)


@register_op("onnx9000.custom", "FakeQuantize")
def fake_quantize(x: Tensor, scale: Tensor, zero_point: Tensor, **kwargs: Any) -> Tensor:
    """Compute Fake Quantization."""
    return record_op("FakeQuantize", [x, scale, zero_point], kwargs)


@register_op("onnx9000.custom", "AllReduce")
def all_reduce(x: Tensor, **kwargs: Any) -> Tensor:
    """Compute Collective AllReduce."""
    return record_op("AllReduce", [x], kwargs)


@register_op("onnx9000.custom", "AllGather")
def all_gather(x: Tensor, **kwargs: Any) -> Tensor:
    """Compute Collective AllGather."""
    return record_op("AllGather", [x], kwargs)


@register_op("onnx9000.custom", "ReduceScatter")
def reduce_scatter(x: Tensor, **kwargs: Any) -> Tensor:
    """Compute Collective ReduceScatter."""
    return record_op("ReduceScatter", [x], kwargs)


@register_op("onnx9000.custom", "Broadcast")
def broadcast(x: Tensor, **kwargs: Any) -> Tensor:
    """Compute Collective Broadcast."""
    return record_op("Broadcast", [x], kwargs)


@register_op("ai.onnx", "LessOrEqual")
def less_or_equal(a: Tensor, b: Tensor) -> Tensor:
    """Compute LessOrEqual."""
    return record_op("LessOrEqual", [a, b])


@register_op("ai.onnx", "GreaterOrEqual")
def greater_or_equal(a: Tensor, b: Tensor) -> Tensor:
    """Compute GreaterOrEqual."""
    return record_op("GreaterOrEqual", [a, b])


@register_op("ai.onnx", "Mean")
def mean(inputs: list[Tensor]) -> Tensor:
    """Compute Mean."""
    return record_op("Mean", inputs)


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


@register_op("ai.onnx", "Multinomial")
def multinomial(input: Tensor, dtype: int = 6, sample_size: int = 1, seed: float = 0.0) -> Tensor:
    """Compute Multinomial."""
    return record_op(
        "Multinomial",
        [input],
        {"dtype": dtype, "sample_size": sample_size, "seed": seed},
    )


@register_op("ai.onnx", "NonMaxSuppression")
def non_max_suppression(
    boxes: Tensor,
    scores: Tensor,
    max_output_boxes_per_class: Optional[Tensor] = None,
    iou_threshold: Optional[Tensor] = None,
    score_threshold: Optional[Tensor] = None,
    center_point_box: int = 0,
) -> Tensor:
    """Compute NonMaxSuppression."""
    inputs = [boxes, scores]
    if max_output_boxes_per_class is not None:
        inputs.append(max_output_boxes_per_class)
    if iou_threshold is not None:
        if max_output_boxes_per_class is None:
            raise ValueError("ONNX sequential inputs rule violation for NonMaxSuppression")
        inputs.append(iou_threshold)
    if score_threshold is not None:
        if iou_threshold is None:
            raise ValueError("ONNX sequential inputs rule violation for NonMaxSuppression")
        inputs.append(score_threshold)
    return record_op("NonMaxSuppression", inputs, {"center_point_box": center_point_box})


@register_op("ai.onnx", "NonZero")
def non_zero(x: Tensor) -> Tensor:
    """Compute NonZero."""
    return record_op("NonZero", [x])


@register_op("ai.onnx", "RandomNormal")
def random_normal(
    dtype: int = 1,
    mean: float = 0.0,
    scale: float = 1.0,
    seed: float = 0.0,
    shape: list[int] = None,
) -> Tensor:
    """Compute RandomNormal."""
    if shape is None:
        shape = []
    return record_op(
        "RandomNormal",
        [],
        {"dtype": dtype, "mean": mean, "scale": scale, "seed": seed, "shape": shape},
    )


@register_op("ai.onnx", "RandomNormalLike")
def random_normal_like(
    input: Tensor,
    dtype: Optional[int] = None,
    mean: float = 0.0,
    scale: float = 1.0,
    seed: float = 0.0,
) -> Tensor:
    """Compute RandomNormalLike."""
    attrs: dict[str, Any] = {"mean": mean, "scale": scale, "seed": seed}
    if dtype is not None:
        attrs["dtype"] = dtype
    return record_op("RandomNormalLike", [input], attrs)


@register_op("ai.onnx", "RandomUniform")
def random_uniform(
    dtype: int = 1,
    high: float = 1.0,
    low: float = 0.0,
    seed: float = 0.0,
    shape: list[int] = None,
) -> Tensor:
    """Compute RandomUniform."""
    if shape is None:
        shape = []
    return record_op(
        "RandomUniform",
        [],
        {"dtype": dtype, "high": high, "low": low, "seed": seed, "shape": shape},
    )


@register_op("ai.onnx", "RandomUniformLike")
def random_uniform_like(
    input: Tensor,
    dtype: Optional[int] = None,
    high: float = 1.0,
    low: float = 0.0,
    seed: float = 0.0,
) -> Tensor:
    """Compute RandomUniformLike."""
    attrs: dict[str, Any] = {"high": high, "low": low, "seed": seed}
    if dtype is not None:
        attrs["dtype"] = dtype
    return record_op("RandomUniformLike", [input], attrs)


@register_op("ai.onnx", "ReduceSumSquare")
def reduce_sum_square(data: Tensor, axes: Optional[list[int]] = None, keepdims: int = 1) -> Tensor:
    """Compute ReduceSumSquare."""
    attrs = {"keepdims": keepdims}
    if axes is not None:
        attrs["axes"] = axes
    return record_op("ReduceSumSquare", [data], attrs)


@register_op("ai.onnx", "ReduceL1")
def reduce_l1(data: Tensor, axes: Optional[list[int]] = None, keepdims: int = 1) -> Tensor:
    """Compute ReduceL1."""
    attrs = {"keepdims": keepdims}
    if axes is not None:
        attrs["axes"] = axes
    return record_op("ReduceL1", [data], attrs)


@register_op("ai.onnx", "ReduceL2")
def reduce_l2(data: Tensor, axes: Optional[list[int]] = None, keepdims: int = 1) -> Tensor:
    """Compute ReduceL2."""
    attrs = {"keepdims": keepdims}
    if axes is not None:
        attrs["axes"] = axes
    return record_op("ReduceL2", [data], attrs)


@register_op("ai.onnx", "ReduceLogSum")
def reduce_log_sum(data: Tensor, axes: Optional[list[int]] = None, keepdims: int = 1) -> Tensor:
    """Compute ReduceLogSum."""
    attrs = {"keepdims": keepdims}
    if axes is not None:
        attrs["axes"] = axes
    return record_op("ReduceLogSum", [data], attrs)


@register_op("ai.onnx", "ReduceLogSumExp")
def reduce_log_sum_exp(data: Tensor, axes: Optional[list[int]] = None, keepdims: int = 1) -> Tensor:
    """Compute ReduceLogSumExp."""
    attrs = {"keepdims": keepdims}
    if axes is not None:
        attrs["axes"] = axes
    return record_op("ReduceLogSumExp", [data], attrs)


@register_op("ai.onnx", "Range")
def range_op(start: Tensor, limit: Tensor, delta: Tensor) -> Tensor:
    """Compute Range."""
    return record_op("Range", [start, limit, delta])


@register_op("ai.onnx", "RegexFullMatch")
def regex_full_match(x: Tensor, pattern: str) -> Tensor:
    """Compute RegexFullMatch."""
    return record_op("RegexFullMatch", [x], {"pattern": pattern})


@register_op("ai.onnx", "Resize")
def resize(
    x: Tensor,
    roi: Optional[Tensor] = None,
    scales: Optional[Tensor] = None,
    sizes: Optional[Tensor] = None,
    **kwargs: Any,
) -> Tensor:
    """Compute Resize."""
    inputs = [x]
    if roi is not None:
        inputs.append(roi)
    elif scales is not None or sizes is not None:
        inputs.append(record_op("Constant", [], {"value": []}))
    if scales is not None:
        inputs.append(scales)
    elif sizes is not None:
        inputs.append(record_op("Constant", [], {"value": []}))
    if sizes is not None:
        inputs.append(sizes)
    return record_op("Resize", inputs, kwargs)


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


@register_op("ai.onnx", "Scatter")
def scatter(data: Tensor, indices: Tensor, updates: Tensor, axis: int = 0) -> Tensor:
    """Compute Scatter."""
    return record_op("Scatter", [data, indices, updates], {"axis": axis})


@register_op("ai.onnx", "ScatterElements")
def scatter_elements(
    data: Tensor,
    indices: Tensor,
    updates: Tensor,
    axis: int = 0,
    reduction: str = "none",
) -> Tensor:
    """Compute ScatterElements."""
    return record_op(
        "ScatterElements",
        [data, indices, updates],
        {"axis": axis, "reduction": reduction},
    )


@register_op("ai.onnx", "ScatterND")
def scatter_nd(data: Tensor, indices: Tensor, updates: Tensor, reduction: str = "none") -> Tensor:
    """Compute ScatterND."""
    return record_op("ScatterND", [data, indices, updates], {"reduction": reduction})


@register_op("ai.onnx", "Shrink")
def shrink(input: Tensor, bias: float = 0.0, lambd: float = 0.5) -> Tensor:
    """Compute Shrink."""
    return record_op("Shrink", [input], {"bias": bias, "lambd": lambd})


@register_op("ai.onnx", "Size")
def size(data: Tensor) -> Tensor:
    """Compute Size."""
    return record_op("Size", [data])


@register_op("ai.onnx", "StringConcat")
def string_concat(x: Tensor, y: Tensor) -> Tensor:
    """Compute StringConcat."""
    return record_op("StringConcat", [x, y])


@register_op("ai.onnx", "StringNormalizer")
def string_normalizer(
    x: Tensor,
    case_change_action: str = "NONE",
    is_case_sensitive: int = 0,
    locale: str = "",
    stopwords: Optional[list[str]] = None,
) -> Tensor:
    """Compute StringNormalizer."""
    attrs: dict[str, Any] = {
        "case_change_action": case_change_action,
        "is_case_sensitive": is_case_sensitive,
        "locale": locale,
    }
    if stopwords is not None:
        attrs["stopwords"] = stopwords
    return record_op("StringNormalizer", [x], attrs)


@register_op("ai.onnx", "StringSplit")
def string_split(x: Tensor, delimiter: str = "", maxsplit: int = -1) -> Tensor:
    """Compute StringSplit."""
    return record_op("StringSplit", [x], {"delimiter": delimiter, "maxsplit": maxsplit})


@register_op("ai.onnx", "Trilu")
def trilu(input: Tensor, k: Optional[Tensor] = None, upper: int = 1) -> Tensor:
    """Compute Trilu."""
    inputs = [input]
    if k is not None:
        inputs.append(k)
    return record_op("Trilu", inputs, {"upper": upper})


@register_op("ai.onnx", "TopK")
def topk(X: Tensor, K: Tensor, axis: int = -1, largest: int = 1, sorted: int = 1) -> Tensor:
    """Compute TopK."""
    return record_op("TopK", [X, K], {"axis": axis, "largest": largest, "sorted": sorted})


@register_op("ai.onnx", "Unique")
def unique(X: Tensor, axis: Optional[int] = None, sorted: int = 1) -> Tensor:
    """Compute Unique."""
    attrs = {"sorted": sorted}
    if axis is not None:
        attrs["axis"] = axis
    return record_op("Unique", [X], attrs)


@register_op("ai.onnx", "SequenceAt")
def sequence_at(input_sequence: Tensor, position: Tensor) -> Tensor:
    """Compute SequenceAt."""
    return record_op("SequenceAt", [input_sequence, position])


@register_op("ai.onnx", "SplitToSequence")
def split_to_sequence(
    input: Tensor, split: Optional[Tensor] = None, axis: int = 0, keepdims: int = 1
) -> Tensor:
    """Compute SplitToSequence."""
    inputs = [input]
    if split is not None:
        inputs.append(split)
    return record_op("SplitToSequence", inputs, {"axis": axis, "keepdims": keepdims})


@register_op("ai.onnx", "SequenceErase")
def sequence_erase(input_sequence: Tensor, position: Optional[Tensor] = None) -> Tensor:
    """Compute SequenceErase."""
    inputs = [input_sequence]
    if position is not None:
        inputs.append(position)
    return record_op("SequenceErase", inputs)


@register_op("ai.onnx", "SequenceLength")
def sequence_length(input_sequence: Tensor) -> Tensor:
    """Compute SequenceLength."""
    return record_op("SequenceLength", [input_sequence])


@register_op("ai.onnx", "AffineGrid")
def affine_grid(theta: Tensor, size: Tensor, align_corners: int = 0) -> Tensor:
    """Compute AffineGrid."""
    return record_op("AffineGrid", [theta, size], {"align_corners": align_corners})


@register_op("ai.onnx", "Argmax")
def argmax(data: Tensor, axis: int = 0, keepdims: int = 1, select_last_index: int = 0) -> Tensor:
    """Compute ArgMax."""
    return record_op(
        "ArgMax",
        [data],
        {"axis": axis, "keepdims": keepdims, "select_last_index": select_last_index},
    )


@register_op("ai.onnx", "Argmin")
def argmin(data: Tensor, axis: int = 0, keepdims: int = 1, select_last_index: int = 0) -> Tensor:
    """Compute ArgMin."""
    return record_op(
        "ArgMin",
        [data],
        {"axis": axis, "keepdims": keepdims, "select_last_index": select_last_index},
    )


@register_op("ai.onnx", "Attention")
def attention(Q: Tensor, K: Tensor, V: Tensor, **kwargs: Any) -> Tensor:
    """Compute Attention."""
    return record_op("Attention", [Q, K, V], kwargs)


@register_op("ai.onnx", "Bernoulli")
def bernoulli(input: Tensor, dtype: Optional[int] = None, seed: float = 0.0) -> Tensor:
    """Compute Bernoulli."""
    attrs: dict[str, Any] = {"seed": seed}
    if dtype is not None:
        attrs["dtype"] = dtype
    return record_op("Bernoulli", [input], attrs)


@register_op("ai.onnx", "CenterCropPad")
def center_crop_pad(input_data: Tensor, shape: Tensor, axes: Optional[list[int]] = None) -> Tensor:
    """Compute CenterCropPad."""
    attrs = {}
    if axes is not None:
        attrs["axes"] = axes
    return record_op("CenterCropPad", [input_data, shape], attrs)


@register_op("ai.onnx", "Constant")
def clip(input: Tensor, min: Optional[Tensor] = None, max: Optional[Tensor] = None) -> Tensor:
    """Compute Clip."""
    inputs = [input]
    if min is not None:
        inputs.append(min)
    elif max is not None:
        inputs.append(record_op("Constant", [], {"value": []}))
    if max is not None:
        inputs.append(max)
    return record_op("Clip", inputs)


@register_op("ai.onnx", "Col2im")
def col2im(
    input: Tensor,
    image_shape: Tensor,
    block_shape: Tensor,
    dilations: Optional[list[int]] = None,
    pads: Optional[list[int]] = None,
    strides: Optional[list[int]] = None,
) -> Tensor:
    """Compute Col2Im."""
    attrs = {}
    if dilations is not None:
        attrs["dilations"] = dilations
    if pads is not None:
        attrs["pads"] = pads
    if strides is not None:
        attrs["strides"] = strides
    return record_op("Col2Im", [input, image_shape, block_shape], attrs)


@register_op("ai.onnx", "Compress")
def compress(input: Tensor, condition: Tensor, axis: Optional[int] = None) -> Tensor:
    """Compute Compress."""
    attrs = {}
    if axis is not None:
        attrs["axis"] = axis
    return record_op("Compress", [input, condition], attrs)


@register_op("ai.onnx", "ConvInteger")
def conv_integer(
    x: Tensor,
    w: Tensor,
    x_zero_point: Optional[Tensor] = None,
    w_zero_point: Optional[Tensor] = None,
    **kwargs: Any,
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


@register_op("ai.onnx", "CumSum")
def cumsum(x: Tensor, axis: Tensor, exclusive: int = 0, reverse: int = 0) -> Tensor:
    """Compute CumSum."""
    return record_op("CumSum", [x, axis], {"exclusive": exclusive, "reverse": reverse})


@register_op("ai.onnx", "Dft")
def dft(
    input: Tensor,
    dft_length: Optional[Tensor] = None,
    axis: int = 1,
    inverse: int = 0,
    onesided: int = 0,
) -> Tensor:
    """Compute DFT."""
    inputs = [input]
    if dft_length is not None:
        inputs.append(dft_length)
    return record_op("DFT", inputs, {"axis": axis, "inverse": inverse, "onesided": onesided})


@register_op("ai.onnx", "Dropout")
def dropout(
    data: Tensor,
    ratio: Optional[Tensor] = None,
    training_mode: Optional[Tensor] = None,
    seed: int = 0,
) -> Tensor:
    """Compute Dropout."""
    inputs = [data]
    if ratio is not None:
        inputs.append(ratio)
    elif training_mode is not None:
        inputs.append(record_op("Constant", [], {"value": []}))
    if training_mode is not None:
        inputs.append(training_mode)
    return record_op("Dropout", inputs, {"seed": seed})


@register_op("ai.onnx", "Reshape")
def reshape(x: Tensor, shape: Tensor) -> Tensor:
    """Reshapes a tensor."""
    return record_op("Reshape", [x, shape])


@register_op("ai.onnx", "AveragePool")
def average_pool(
    x: Tensor,
    kernel_shape: list[int],
    strides: Optional[list[int]] = None,
    pads: Optional[list[int]] = None,
) -> Tensor:
    """Compute AveragePool."""
    attr = {"kernel_shape": kernel_shape}
    if strides:
        attr["strides"] = strides
    if pads:
        attr["pads"] = pads
    return record_op("AveragePool", [x], attr)


@register_op("ai.onnx", "MaxPool")
def max_pool(
    x: Tensor,
    kernel_shape: list[int],
    strides: Optional[list[int]] = None,
    pads: Optional[list[int]] = None,
) -> Tensor:
    """Compute MaxPool."""
    attr = {"kernel_shape": kernel_shape}
    if strides:
        attr["strides"] = strides
    if pads:
        attr["pads"] = pads
    return record_op("MaxPool", [x], attr)


@register_op("ai.onnx", "GlobalAveragePool")
def global_average_pool(x: Tensor) -> Tensor:
    """Compute GlobalAveragePool."""
    return record_op("GlobalAveragePool", [x])


@register_op("ai.onnx", "GlobalMaxPool")
def global_max_pool(x: Tensor) -> Tensor:
    """Compute GlobalMaxPool."""
    return record_op("GlobalMaxPool", [x])


@register_op("ai.onnx", "Transpose")
def transpose(x: Tensor, perm: Optional[list[int]] = None) -> Tensor:
    """Transposes a tensor."""
    attributes = {"perm": perm} if perm is not None else {}
    return record_op("Transpose", [x], attributes)


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
    c: Optional[Tensor] = None,
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


@register_op("ai.onnx", "ReduceSum")
def reduce_sum(x: Tensor, axes: Optional[list[int]] = None, keepdims: bool = True) -> Tensor:
    """Compute the sum of the input tensor's element along the provided axes."""
    attributes: dict[str, Any] = {"keepdims": 1 if keepdims else 0}
    if axes is not None:
        attributes["axes"] = axes
    return record_op("ReduceSum", [x], attributes)


@register_op("ai.onnx", "ReduceMean")
def reduce_mean(x: Tensor, axes: Optional[list[int]] = None, keepdims: bool = True) -> Tensor:
    """Compute the mean of the input tensor's element along the provided axes."""
    attributes: dict[str, Any] = {"keepdims": 1 if keepdims else 0}
    if axes is not None:
        attributes["axes"] = axes
    return record_op("ReduceMean", [x], attributes)


@register_op("ai.onnx", "ReduceMax")
def reduce_max(x: Tensor, axes: Optional[list[int]] = None, keepdims: bool = True) -> Tensor:
    """Compute ReduceMax."""
    attributes: dict[str, Any] = {"keepdims": 1 if keepdims else 0}
    if axes is not None:
        attributes["axes"] = axes
    return record_op("ReduceMax", [x], attributes)


@register_op("ai.onnx", "ReduceMin")
def reduce_min(x: Tensor, axes: Optional[list[int]] = None, keepdims: bool = True) -> Tensor:
    """Compute ReduceMin."""
    attributes: dict[str, Any] = {"keepdims": 1 if keepdims else 0}
    if axes is not None:
        attributes["axes"] = axes
    return record_op("ReduceMin", [x], attributes)


@register_op("ai.onnx", "ReduceProd")
def reduce_prod(x: Tensor, axes: Optional[list[int]] = None, keepdims: bool = True) -> Tensor:
    """Compute ReduceProd."""
    attributes: dict[str, Any] = {"keepdims": 1 if keepdims else 0}
    if axes is not None:
        attributes["axes"] = axes
    return record_op("ReduceProd", [x], attributes)


@register_op("ai.onnx", "Conv")
def conv(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
    strides: Optional[list[int]] = None,
    pads: Optional[list[int]] = None,
) -> Tensor:
    """Compute an N-D convolution."""
    if strides is None:
        strides = [1, 1]
    if pads is None:
        pads = [0, 0, 0, 0]
    inputs = [x, w]
    if b is not None:
        inputs.append(b)
    attributes: dict[str, Any] = {"strides": strides, "pads": pads}
    return record_op("Conv", inputs, attributes)


@register_op("ai.onnx", "DynamicQuantizeLinear")
def dynamic_quantize_linear(x: Tensor) -> Any:
    """Compute DynamicQuantizeLinear."""
    return record_op("DynamicQuantizeLinear", [x])


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


@register_op("ai.onnx", "Expand")
def expand(input: Tensor, shape: Tensor) -> Tensor:
    """Execute the Expand process and return the computed results."""
    return record_op("Expand", [input, shape])


@register_op("ai.onnx", "Slice")
def slice(
    data: Tensor,
    starts: Tensor,
    ends: Tensor,
    axes: Tensor = None,
    steps: Tensor = None,
) -> Tensor:
    """Compute Slice."""
    inputs = [data, starts, ends]
    if axes is not None:
        inputs.append(axes)
        if steps is not None:
            inputs.append(steps)
    return record_op("Slice", inputs)


@register_op("ai.onnx", "Tile")
def tile(input: Tensor, repeats: Tensor) -> Tensor:
    """Compute Tile."""
    return record_op("Tile", [input, repeats])


@register_op("ai.onnx", "Gather")
def gather(data: Tensor, indices: Tensor, axis: int = 0) -> Tensor:
    """Compute Gather."""
    return record_op("Gather", [data, indices], {"axis": axis})


@register_op("ai.onnx", "GatherND")
def gather_nd(data: Tensor, indices: Tensor, batch_dims: int = 0) -> Tensor:
    """Compute GatherND."""
    return record_op("GatherND", [data, indices], {"batch_dims": batch_dims})


@register_op("ai.onnx", "GatherElements")
def gather_elements(data: Tensor, indices: Tensor, axis: int = 0) -> Tensor:
    """Compute GatherElements."""
    return record_op("GatherElements", [data, indices], {"axis": axis})


@register_op("ai.onnx", "DepthToSpace")
def depth_to_space(data: Tensor, blocksize: int, mode: str = "DCR") -> Tensor:
    """Compute DepthToSpace."""
    return record_op("DepthToSpace", [data], {"blocksize": blocksize, "mode": mode})


@register_op("ai.onnx", "SpaceToDepth")
def space_to_depth(data: Tensor, blocksize: int) -> Tensor:
    """Compute SpaceToDepth."""
    return record_op("SpaceToDepth", [data], {"blocksize": blocksize})


@register_op("ai.onnx", "Sum")
def sum(data: list[Tensor]) -> Tensor:
    """Compute Sum."""
    return record_op("Sum", data)


@register_op("ai.onnx", "Swish")
def swish(x: Tensor) -> Tensor:
    """Compute Swish."""
    return record_op("Swish", [x])


@register_op("ai.onnx", "OneHot")
def one_hot(indices: Tensor, depth: Tensor, values: Tensor, axis: int = -1) -> Tensor:
    """Compute OneHot."""
    return record_op("OneHot", [indices, depth, values], {"axis": axis})


@register_op("ai.onnx", "LRN")
def lrn(
    x: Tensor,
    alpha: float = 0.0001,
    beta: float = 0.75,
    bias: float = 1.0,
    size: int = 1,
) -> Tensor:
    """Compute LRN."""
    return record_op("LRN", [x], {"alpha": alpha, "beta": beta, "bias": bias, "size": size})


@register_op("ai.onnx", "GroupNormalization")
def group_normalization(
    x: Tensor, scale: Tensor, b: Tensor, epsilon: float = 1e-05, num_groups: int = 1
) -> Tensor:
    """Compute GroupNormalization."""
    return record_op(
        "GroupNormalization",
        [x, scale, b],
        {"epsilon": epsilon, "num_groups": num_groups},
    )


@register_op("ai.onnx", "BatchNormalization")
def batch_normalization(x: Tensor) -> Tensor:
    """Compute BatchNormalization."""
    return record_op("BatchNormalization", [x])


@register_op("ai.onnx", "BitCast")
def bit_cast(x: Tensor) -> Tensor:
    """Compute BitCast."""
    return record_op("BitCast", [x])


@register_op("ai.onnx", "CumProd")
def cum_prod(x: Tensor) -> Tensor:
    """Compute CumProd."""
    return record_op("CumProd", [x])


@register_op("ai.onnx", "Flatten")
def flatten(x: Tensor) -> Tensor:
    """Compute Flatten."""
    return record_op("Flatten", [x])


@register_op("ai.onnx", "Gelu")
def gelu(x: Tensor) -> Tensor:
    """Compute Gelu."""
    return record_op("Gelu", [x])


@register_op("ai.onnx", "GlobalLpPool")
def global_lp_pool(x: Tensor) -> Tensor:
    """Compute GlobalLpPool."""
    return record_op("GlobalLpPool", [x])


@register_op("ai.onnx", "GridSample")
def grid_sample(x: Tensor) -> Tensor:
    """Compute GridSample."""
    return record_op("GridSample", [x])


@register_op("ai.onnx", "HammingWindow")
def hamming_window(x: Tensor) -> Tensor:
    """Compute HammingWindow."""
    return record_op("HammingWindow", [x])


@register_op("ai.onnx", "HannWindow")
def hann_window(x: Tensor) -> Tensor:
    """Compute HannWindow."""
    return record_op("HannWindow", [x])


@register_op("ai.onnx", "Identity")
def identity(x: Tensor) -> Tensor:
    """Compute Identity."""
    return record_op("Identity", [x])


@register_op("ai.onnx", "If")
def if_op(x: Tensor) -> Tensor:
    """Compute If."""
    return record_op("If", [x])


@register_op("ai.onnx", "ImageDecoder")
def image_decoder(x: Tensor) -> Tensor:
    """Compute ImageDecoder."""
    return record_op("ImageDecoder", [x])


@register_op("ai.onnx", "Log")
def log(x: Tensor) -> Tensor:
    """Compute Log."""
    return record_op("Log", [x])


@register_op("ai.onnx", "Loop")
def loop(x: Tensor) -> Tensor:
    """Compute Loop."""
    return record_op("Loop", [x])


@register_op("ai.onnx", "MatMulInteger")
def mat_mul_integer(x: Tensor) -> Tensor:
    """Compute MatMulInteger."""
    return record_op("MatMulInteger", [x])


@register_op("ai.onnx", "MaxRoiPool")
def max_roi_pool(x: Tensor) -> Tensor:
    """Compute MaxRoiPool."""
    return record_op("MaxRoiPool", [x])


@register_op("ai.onnx", "MaxUnpool")
def max_unpool(x: Tensor) -> Tensor:
    """Compute MaxUnpool."""
    return record_op("MaxUnpool", [x])


@register_op("ai.onnx", "NegativeLogLikelihoodLoss")
def negative_log_likelihood_loss(x: Tensor) -> Tensor:
    """Compute NegativeLogLikelihoodLoss."""
    return record_op("NegativeLogLikelihoodLoss", [x])


@register_op("ai.onnx", "Optional")
def optional(x: Tensor) -> Tensor:
    """Compute Optional."""
    return record_op("Optional", [x])


@register_op("ai.onnx", "OptionalGetElement")
def optional_get_element(x: Tensor) -> Tensor:
    """Compute OptionalGetElement."""
    return record_op("OptionalGetElement", [x])


@register_op("ai.onnx", "OptionalHasElement")
def optional_has_element(x: Tensor) -> Tensor:
    """Compute OptionalHasElement."""
    return record_op("OptionalHasElement", [x])


@register_op("ai.onnx", "Pad")
def pad(x: Tensor) -> Tensor:
    """Compute Pad."""
    return record_op("Pad", [x])


@register_op("ai.onnx", "QLinearConv")
def q_linear_conv(x: Tensor) -> Tensor:
    """Compute QLinearConv."""
    return record_op("QLinearConv", [x])


@register_op("ai.onnx", "QLinearMatMul")
def q_linear_mat_mul(x: Tensor) -> Tensor:
    """Compute QLinearMatMul."""
    return record_op("QLinearMatMul", [x])


@register_op("ai.onnx", "RMSNormalization")
def rms_normalization(x: Tensor, scale: Tensor) -> Tensor:
    """Compute RMSNormalization."""
    return record_op("RMSNormalization", [x, scale])


@register_op("ai.onnx", "Range")
def range_op2(x: Tensor) -> Tensor:
    """Compute Range."""
    return record_op("Range", [x])


@register_op("ai.onnx", "RoiAlign")
def roi_align(x: Tensor) -> Tensor:
    """Compute RoiAlign."""
    return record_op("RoiAlign", [x])


@register_op("ai.onnx", "RotaryEmbedding")
def rotary_embedding(x: Tensor) -> Tensor:
    """Compute RotaryEmbedding."""
    return record_op("RotaryEmbedding", [x])


@register_op("ai.onnx", "Scan")
def scan(x: Tensor) -> Tensor:
    """Compute Scan."""
    return record_op("Scan", [x])


@register_op("ai.onnx", "Shape")
def shape(x: Tensor) -> Tensor:
    """Compute Shape."""
    return record_op("Shape", [x])


@register_op("ai.onnx", "SoftmaxCrossEntropyLoss")
def softmax_cross_entropy_loss(x: Tensor) -> Tensor:
    """Compute SoftmaxCrossEntropyLoss."""
    return record_op("SoftmaxCrossEntropyLoss", [x])


@register_op("ai.onnx", "Split")
def split(x: Tensor) -> Tensor:
    """Compute Split."""
    return record_op("Split", [x])


@register_op("ai.onnx", "Sqrt")
def sqrt(x: Tensor) -> Tensor:
    """Compute Sqrt."""
    return record_op("Sqrt", [x])


@register_op("ai.onnx", "Squeeze")
def squeeze(x: Tensor) -> Tensor:
    """Compute Squeeze."""
    return record_op("Squeeze", [x])


@register_op("ai.onnx", "TensorScatter")
def tensor_scatter(x: Tensor) -> Tensor:
    """Compute TensorScatter."""
    return record_op("TensorScatter", [x])


@register_op("ai.onnx", "TfIdfVectorizer")
def tf_idf_vectorizer(x: Tensor) -> Tensor:
    """Compute TfIdfVectorizer."""
    return record_op("TfIdfVectorizer", [x])


@register_op("ai.onnx", "Unsqueeze")
def unsqueeze(x: Tensor) -> Tensor:
    """Compute Unsqueeze."""
    return record_op("Unsqueeze", [x])


@register_op("ai.onnx", "Upsample")
def upsample(x: Tensor) -> Tensor:
    """Compute Upsample."""
    return record_op("Upsample", [x])


@register_op("ai.onnx", "Xor")
def xor(x: Tensor) -> Tensor:
    """Compute Xor."""
    return record_op("Xor", [x])


@register_op("ai.onnx", "STFT")
def stft(
    signal: Tensor,
    frame_step: Optional[Tensor] = None,
    window_length: Optional[Tensor] = None,
    frame_length: Optional[Tensor] = None,
    window: Optional[Tensor] = None,
    **kwargs: Any,
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
    frame_step: Optional[Tensor] = None,
    window_length: Optional[Tensor] = None,
    frame_length: Optional[Tensor] = None,
    window: Optional[Tensor] = None,
    **kwargs: Any,
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


@register_op("ai.onnx", "DeformConv")
def deform_conv(
    x: Tensor,
    w: Tensor,
    offset: Tensor,
    b: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    **kwargs: Any,
) -> Tensor:
    """Compute Deformable Convolution."""
    inputs = [x, w, offset]
    if b is not None:
        inputs.append(b)
    if mask is not None:
        if b is None:
            raise ValueError("Mask provided without Bias. ONNX requires sequential inputs.")
        inputs.append(mask)
    return record_op("DeformConv", inputs, kwargs)


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


@register_op("ai.onnx", "Im2Col")
def im2col(x: Tensor) -> Tensor:
    """Compute Im2Col."""
    return record_op("Im2Col", [x])


@register_op("ai.onnx", "Repeat")
def repeat(x: Tensor, repeats: Tensor) -> Tensor:
    """Compute Repeat."""
    return record_op("Repeat", [x, repeats])


@register_op("ai.onnx", "Conv1D")
def conv1d(x: Tensor, w: Tensor, b: Optional[Tensor] = None, **kwargs: Any) -> Tensor:
    """Compute Conv1D."""
    return record_op("Conv1D", [x, w] + ([b] if b else []))


@register_op("ai.onnx", "Conv2D")
def conv2d(x: Tensor, w: Tensor, b: Optional[Tensor] = None, **kwargs: Any) -> Tensor:
    """Compute Conv2D."""
    return record_op("Conv2D", [x, w] + ([b] if b else []))


@register_op("ai.onnx", "Conv3D")
def conv3d(x: Tensor, w: Tensor, b: Optional[Tensor] = None, **kwargs: Any) -> Tensor:
    """Compute Conv3D."""
    return record_op("Conv3D", [x, w] + ([b] if b else []))


@register_op("ai.onnx", "ConvTranspose1D")
def conv_transpose1d(x: Tensor, w: Tensor, b: Optional[Tensor] = None, **kwargs: Any) -> Tensor:
    """Compute ConvTranspose1D."""
    return record_op("ConvTranspose1D", [x, w] + ([b] if b else []))


@register_op("ai.onnx", "ConvTranspose2D")
def conv_transpose2d(x: Tensor, w: Tensor, b: Optional[Tensor] = None, **kwargs: Any) -> Tensor:
    """Compute ConvTranspose2D."""
    return record_op("ConvTranspose2D", [x, w] + ([b] if b else []))


@register_op("ai.onnx", "ConvTranspose3D")
def conv_transpose3d(x: Tensor, w: Tensor, b: Optional[Tensor] = None, **kwargs: Any) -> Tensor:
    """Compute ConvTranspose3D."""
    return record_op("ConvTranspose3D", [x, w] + ([b] if b else []))


@register_op("ai.onnx", "DepthwiseConv2D")
def depthwise_conv2d(x: Tensor, w: Tensor, b: Optional[Tensor] = None, **kwargs: Any) -> Tensor:
    """Compute DepthwiseConv2D."""
    return record_op("DepthwiseConv2D", [x, w] + ([b] if b else []))


@register_op("ai.onnx", "DeformableConv2D")
def deformable_conv2d(
    x: Tensor, w: Tensor, offset: Tensor, b: Optional[Tensor] = None, **kwargs: Any
) -> Tensor:
    """Compute DeformableConv2D."""
    return record_op("DeformableConv2D", [x, w, offset] + ([b] if b else []))


@register_op("ai.onnx", "MaxPool1D")
def max_pool1d(x: Tensor, **kwargs: Any) -> Tensor:
    """Compute MaxPool1D."""
    return record_op("MaxPool1D", [x])


@register_op("ai.onnx", "MaxPool2D")
def max_pool2d(x: Tensor, **kwargs: Any) -> Tensor:
    """Compute MaxPool2D."""
    return record_op("MaxPool2D", [x])


@register_op("ai.onnx", "MaxPool3D")
def max_pool3d(x: Tensor, **kwargs: Any) -> Tensor:
    """Compute MaxPool3D."""
    return record_op("MaxPool3D", [x])


@register_op("ai.onnx", "AveragePool1D")
def average_pool1d(x: Tensor, **kwargs: Any) -> Tensor:
    """Compute AveragePool1D."""
    return record_op("AveragePool1D", [x])


@register_op("ai.onnx", "AveragePool2D")
def average_pool2d(x: Tensor, **kwargs: Any) -> Tensor:
    """Compute AveragePool2D."""
    return record_op("AveragePool2D", [x])


@register_op("ai.onnx", "AveragePool3D")
def average_pool3d(x: Tensor, **kwargs: Any) -> Tensor:
    """Compute AveragePool3D."""
    return record_op("AveragePool3D", [x])


@register_op("ai.onnx", "AdaptiveMaxPool2D")
def adaptive_max_pool2d(x: Tensor, **kwargs: Any) -> Tensor:
    """Compute AdaptiveMaxPool2D."""
    return record_op("AdaptiveMaxPool2D", [x])


@register_op("ai.onnx", "AdaptiveAvgPool2D")
def adaptive_avg_pool2d(x: Tensor, **kwargs: Any) -> Tensor:
    """Compute AdaptiveAvgPool2D."""
    return record_op("AdaptiveAvgPool2D", [x])


@register_op("ai.onnx", "LocalResponseNorm")
def local_response_norm(
    x: Tensor, size: int, alpha: float = 0.0001, beta: float = 0.75, bias: float = 1.0
) -> Tensor:
    """Compute LocalResponseNorm."""
    return record_op("LocalResponseNorm", [x])


@register_op("ai.onnx", "AdaLN")
def adaln(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Compute AdaLN."""
    return record_op("AdaLN", [x, shift, scale])


@register_op("ai.onnx", "Silu")
def silu(x: Tensor) -> Tensor:
    """Compute Silu."""
    return record_op("Silu", [x])


@register_op("ai.onnx", "SwiGLU")
def swiglu(x: Tensor, y: Tensor) -> Tensor:
    """Compute SwiGLU."""
    return record_op("SwiGLU", [x, y])


@register_op("ai.onnx", "GeGLU")
def geglu(x: Tensor, y: Tensor) -> Tensor:
    """Compute GeGLU."""
    return record_op("GeGLU", [x, y])


@register_op("ai.onnx", "ReGLU")
def reglu(x: Tensor, y: Tensor) -> Tensor:
    """Compute ReGLU."""
    return record_op("ReGLU", [x, y])


@register_op("ai.onnx", "MultiHeadAttention")
def multi_head_attention(query: Tensor, key: Tensor, value: Tensor, **kwargs: Any) -> Tensor:
    """Compute MultiHeadAttention."""
    return record_op("MultiHeadAttention", [query, key, value])


@register_op("ai.onnx", "GroupedQueryAttention")
def grouped_query_attention(query: Tensor, key: Tensor, value: Tensor, **kwargs: Any) -> Tensor:
    """Compute GroupedQueryAttention."""
    return record_op("GroupedQueryAttention", [query, key, value])


@register_op("ai.onnx", "MultiQueryAttention")
def multi_query_attention(query: Tensor, key: Tensor, value: Tensor, **kwargs: Any) -> Tensor:
    """Compute MultiQueryAttention."""
    return record_op("MultiQueryAttention", [query, key, value])


@register_op("ai.onnx", "FlashAttention")
def flash_attention(query: Tensor, key: Tensor, value: Tensor, **kwargs: Any) -> Tensor:
    """Compute FlashAttention."""
    return record_op("FlashAttention", [query, key, value])


@register_op("ai.onnx", "PagedAttention")
def paged_attention(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
    **kwargs: Any,
) -> Tensor:
    """Compute PagedAttention."""
    return record_op("PagedAttention", [query, key_cache, value_cache, block_tables, context_lens])


@register_op("ai.onnx", "RoPE1D")
def rope1d(x: Tensor, freqs: Tensor, **kwargs: Any) -> Tensor:
    """Compute RoPE1D."""
    return record_op("RoPE1D", [x, freqs])


@register_op("ai.onnx", "RoPE2D")
def rope2d(x: Tensor, freqs: Tensor, **kwargs: Any) -> Tensor:
    """Compute RoPE2D."""
    return record_op("RoPE2D", [x, freqs])


@register_op("ai.onnx", "RoPE3D")
def rope3d(x: Tensor, freqs: Tensor, **kwargs: Any) -> Tensor:
    """Compute RoPE3D."""
    return record_op("RoPE3D", [x, freqs])


@register_op("ai.onnx", "ALiBi")
def alibi(x: Tensor, **kwargs: Any) -> Tensor:
    """Compute ALiBi."""
    return record_op("ALiBi", [x])


@register_op("ai.onnx", "SlidingWindowAttention")
def sliding_window_attention(
    query: Tensor, key: Tensor, value: Tensor, window_size: int, **kwargs: Any
) -> Tensor:
    """Compute SlidingWindowAttention."""
    return record_op("SlidingWindowAttention", [query, key, value])


@register_op("ai.onnx", "StateSpaceModel")
def state_space_model(x: Tensor, a: Tensor, b: Tensor, c: Tensor, **kwargs: Any) -> Tensor:
    """Compute StateSpaceModel."""
    return record_op("StateSpaceModel", [x, a, b, c])


@register_op("ai.onnx", "QuantizeLinear")
def quantize_linear(
    x: Tensor, y_scale: Tensor, y_zero_point: Optional[Tensor] = None, **kwargs: Any
) -> Tensor:
    """Compute QuantizeLinear."""
    return record_op("QuantizeLinear", [x, y_scale] + ([y_zero_point] if y_zero_point else []))


@register_op("ai.onnx", "DequantizeLinear")
def dequantize_linear(
    x: Tensor, x_scale: Tensor, x_zero_point: Optional[Tensor] = None, **kwargs: Any
) -> Tensor:
    """Compute DequantizeLinear."""
    return record_op("DequantizeLinear", [x, x_scale] + ([x_zero_point] if x_zero_point else []))
