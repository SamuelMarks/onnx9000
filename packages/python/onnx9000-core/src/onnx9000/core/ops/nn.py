"""Neural network, activation, convolution, pooling, normalization, and attention ONNX operators."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.core.ir import AttributeValue, Tensor
from onnx9000.core.ops.base import record_op
from onnx9000.core.registry import register_op


@register_op("ai.onnx", "Relu")
def relu(x: Tensor) -> Tensor:
    """Compute Rectified Linear Unit."""
    return record_op("Relu", [x])


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


@register_op("ai.onnx", "ConvTranspose")
def conv_transpose(x: Tensor, w: Tensor, b: Tensor = None, **kwargs: AttributeValue) -> Tensor:
    """Compute ConvTranspose."""
    inputs = [x, w]
    if b is not None:
        inputs.append(b)
    return record_op("ConvTranspose", inputs, kwargs)


@register_op("ai.onnx", "LpNormalization")
def lp_normalization(input: Tensor, axis: int = -1, p: int = 2) -> Tensor:
    """Compute LpNormalization."""
    return record_op("LpNormalization", [input], {"axis": axis, "p": p})


@register_op("ai.onnx", "LpPool")
def lp_pool(
    input: Tensor, kernel_shape: list[int], p: float = 2.0, **kwargs: AttributeValue
) -> Tensor:
    """Compute LpPool."""
    return record_op("LpPool", [input], {"kernel_shape": kernel_shape, "p": p, **kwargs})


@register_op("ai.onnx", "LayerNormalization")
def layer_normalization(
    x: Tensor,
    scale: Tensor,
    b: Tensor = None,
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
def kv_cache(cache: Tensor, new_key: Tensor, new_value: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute Key-Value Cache update."""
    return record_op("KVCache", [cache, new_key, new_value], kwargs)


@register_op("onnx9000.custom", "RoPE")
def rope(x: Tensor, cos: Tensor, sin: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute Rotary Positional Embeddings."""
    return record_op("RoPE", [x, cos, sin], kwargs)


@register_op("onnx9000.custom", "DeformConv")
def deform_conv_custom(
    x: Tensor, offset: Tensor, w: Tensor, b: Tensor = None, **kwargs: AttributeValue
) -> Tensor:
    """Compute Deformable Convolution."""
    inputs = [x, offset, w]
    if b is not None:
        inputs.append(b)
    return record_op("DeformConv", inputs, kwargs)


@register_op("ai.onnx", "Shrink")
def shrink(input: Tensor, bias: float = 0.0, lambd: float = 0.5) -> Tensor:
    """Compute Shrink."""
    return record_op("Shrink", [input], {"bias": bias, "lambd": lambd})


@register_op("ai.onnx", "AffineGrid")
def affine_grid(theta: Tensor, size: Tensor, align_corners: int = 0) -> Tensor:
    """Compute AffineGrid."""
    return record_op("AffineGrid", [theta, size], {"align_corners": align_corners})


@register_op("ai.onnx", "Attention")
def attention(Q: Tensor, K: Tensor, V: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute Attention."""
    return record_op("Attention", [Q, K, V], kwargs)


@register_op("ai.onnx", "Col2im")
def col2im(
    input: Tensor,
    image_shape: Tensor,
    block_shape: Tensor,
    dilations: list[int] | None = None,
    pads: list[int] | None = None,
    strides: list[int] | None = None,
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


@register_op("ai.onnx", "Dropout")
def dropout(
    data: Tensor,
    ratio: Tensor = None,
    training_mode: Tensor = None,
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


@register_op("ai.onnx", "AveragePool")
def average_pool(
    x: Tensor,
    kernel_shape: list[int],
    strides: list[int] | None = None,
    pads: list[int] | None = None,
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
    strides: list[int] | None = None,
    pads: list[int] | None = None,
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


@register_op("ai.onnx", "Conv")
def conv(
    x: Tensor,
    w: Tensor,
    b: Tensor = None,
    strides: list[int] | None = None,
    pads: list[int] | None = None,
) -> Tensor:
    """Compute an N-D convolution."""
    if strides is None:
        strides = [1, 1]
    if pads is None:
        pads = [0, 0, 0, 0]
    inputs = [x, w]
    if b is not None:
        inputs.append(b)
    attributes: dict[str, AttributeValue] = {"strides": strides, "pads": pads}
    return record_op("Conv", inputs, attributes)


@register_op("ai.onnx", "Swish")
def swish(x: Tensor) -> Tensor:
    """Compute Swish."""
    return record_op("Swish", [x])


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


@register_op("ai.onnx", "RMSNormalization")
def rms_normalization(x: Tensor, scale: Tensor) -> Tensor:
    """Compute RMSNormalization."""
    return record_op("RMSNormalization", [x, scale])


@register_op("ai.onnx", "RoiAlign")
def roi_align(x: Tensor) -> Tensor:
    """Compute RoiAlign."""
    return record_op("RoiAlign", [x])


@register_op("ai.onnx", "RotaryEmbedding")
def rotary_embedding(x: Tensor) -> Tensor:
    """Compute RotaryEmbedding."""
    return record_op("RotaryEmbedding", [x])


@register_op("ai.onnx", "SoftmaxCrossEntropyLoss")
def softmax_cross_entropy_loss(x: Tensor) -> Tensor:
    """Compute SoftmaxCrossEntropyLoss."""
    return record_op("SoftmaxCrossEntropyLoss", [x])


@register_op("ai.onnx", "DeformConv")
def deform_conv(
    x: Tensor,
    w: Tensor,
    offset: Tensor,
    b: Tensor = None,
    mask: Tensor = None,
    **kwargs: AttributeValue,
) -> Tensor:
    """Compute Deformable Convolution."""
    inputs = [x, w, offset]
    if b is not None:
        inputs.append(b)
    if mask is not None:
        if b is None:
            raise Onnx9000Error("Mask provided without Bias. ONNX requires sequential inputs.")
        inputs.append(mask)
    return record_op("DeformConv", inputs, kwargs)


@register_op("ai.onnx", "Im2Col")
def im2col(x: Tensor) -> Tensor:
    """Compute Im2Col."""
    return record_op("Im2Col", [x])


@register_op("ai.onnx", "Conv1D")
def conv1d(x: Tensor, w: Tensor, b: Tensor = None, **kwargs: AttributeValue) -> Tensor:
    """Compute Conv1D."""
    return record_op("Conv1D", [x, w] + ([b] if b else []))


@register_op("ai.onnx", "Conv2D")
def conv2d(x: Tensor, w: Tensor, b: Tensor = None, **kwargs: AttributeValue) -> Tensor:
    """Compute Conv2D."""
    return record_op("Conv2D", [x, w] + ([b] if b else []))


@register_op("ai.onnx", "Conv3D")
def conv3d(x: Tensor, w: Tensor, b: Tensor = None, **kwargs: AttributeValue) -> Tensor:
    """Compute Conv3D."""
    return record_op("Conv3D", [x, w] + ([b] if b else []))


@register_op("ai.onnx", "ConvTranspose1D")
def conv_transpose1d(x: Tensor, w: Tensor, b: Tensor = None, **kwargs: AttributeValue) -> Tensor:
    """Compute ConvTranspose1D."""
    return record_op("ConvTranspose1D", [x, w] + ([b] if b else []))


@register_op("ai.onnx", "ConvTranspose2D")
def conv_transpose2d(x: Tensor, w: Tensor, b: Tensor = None, **kwargs: AttributeValue) -> Tensor:
    """Compute ConvTranspose2D."""
    return record_op("ConvTranspose2D", [x, w] + ([b] if b else []))


@register_op("ai.onnx", "ConvTranspose3D")
def conv_transpose3d(x: Tensor, w: Tensor, b: Tensor = None, **kwargs: AttributeValue) -> Tensor:
    """Compute ConvTranspose3D."""
    return record_op("ConvTranspose3D", [x, w] + ([b] if b else []))


@register_op("ai.onnx", "DepthwiseConv2D")
def depthwise_conv2d(x: Tensor, w: Tensor, b: Tensor = None, **kwargs: AttributeValue) -> Tensor:
    """Compute DepthwiseConv2D."""
    return record_op("DepthwiseConv2D", [x, w] + ([b] if b else []))


@register_op("ai.onnx", "DeformableConv2D")
def deformable_conv2d(
    x: Tensor, w: Tensor, offset: Tensor, b: Tensor = None, **kwargs: AttributeValue
) -> Tensor:
    """Compute DeformableConv2D."""
    return record_op("DeformableConv2D", [x, w, offset] + ([b] if b else []))


@register_op("ai.onnx", "MaxPool1D")
def max_pool1d(x: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute MaxPool1D."""
    return record_op("MaxPool1D", [x])


@register_op("ai.onnx", "MaxPool2D")
def max_pool2d(x: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute MaxPool2D."""
    return record_op("MaxPool2D", [x])


@register_op("ai.onnx", "MaxPool3D")
def max_pool3d(x: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute MaxPool3D."""
    return record_op("MaxPool3D", [x])


@register_op("ai.onnx", "AveragePool1D")
def average_pool1d(x: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute AveragePool1D."""
    return record_op("AveragePool1D", [x])


@register_op("ai.onnx", "AveragePool2D")
def average_pool2d(x: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute AveragePool2D."""
    return record_op("AveragePool2D", [x])


@register_op("ai.onnx", "AveragePool3D")
def average_pool3d(x: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute AveragePool3D."""
    return record_op("AveragePool3D", [x])


@register_op("ai.onnx", "AdaptiveMaxPool2D")
def adaptive_max_pool2d(x: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute AdaptiveMaxPool2D."""
    return record_op("AdaptiveMaxPool2D", [x])


@register_op("ai.onnx", "AdaptiveAvgPool2D")
def adaptive_avg_pool2d(x: Tensor, **kwargs: AttributeValue) -> Tensor:
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
def multi_head_attention(
    query: Tensor, key: Tensor, value: Tensor, **kwargs: AttributeValue
) -> Tensor:
    """Compute MultiHeadAttention."""
    return record_op("MultiHeadAttention", [query, key, value])


@register_op("ai.onnx", "GroupedQueryAttention")
def grouped_query_attention(
    query: Tensor, key: Tensor, value: Tensor, **kwargs: AttributeValue
) -> Tensor:
    """Compute GroupedQueryAttention."""
    return record_op("GroupedQueryAttention", [query, key, value])


@register_op("ai.onnx", "MultiQueryAttention")
def multi_query_attention(
    query: Tensor, key: Tensor, value: Tensor, **kwargs: AttributeValue
) -> Tensor:
    """Compute MultiQueryAttention."""
    return record_op("MultiQueryAttention", [query, key, value])


@register_op("ai.onnx", "FlashAttention")
def flash_attention(query: Tensor, key: Tensor, value: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute FlashAttention."""
    return record_op("FlashAttention", [query, key, value])


@register_op("ai.onnx", "PagedAttention")
def paged_attention(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
    **kwargs: AttributeValue,
) -> Tensor:
    """Compute PagedAttention."""
    return record_op("PagedAttention", [query, key_cache, value_cache, block_tables, context_lens])


@register_op("ai.onnx", "RoPE1D")
def rope1d(x: Tensor, freqs: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute RoPE1D."""
    return record_op("RoPE1D", [x, freqs])


@register_op("ai.onnx", "RoPE2D")
def rope2d(x: Tensor, freqs: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute RoPE2D."""
    return record_op("RoPE2D", [x, freqs])


@register_op("ai.onnx", "RoPE3D")
def rope3d(x: Tensor, freqs: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute RoPE3D."""
    return record_op("RoPE3D", [x, freqs])


@register_op("ai.onnx", "ALiBi")
def alibi(x: Tensor, **kwargs: AttributeValue) -> Tensor:
    """Compute ALiBi."""
    return record_op("ALiBi", [x])


@register_op("ai.onnx", "SlidingWindowAttention")
def sliding_window_attention(
    query: Tensor, key: Tensor, value: Tensor, window_size: int, **kwargs: AttributeValue
) -> Tensor:
    """Compute SlidingWindowAttention."""
    return record_op("SlidingWindowAttention", [query, key, value])


@register_op("ai.onnx", "StateSpaceModel")
def state_space_model(
    x: Tensor, a: Tensor, b: Tensor, c: Tensor, **kwargs: AttributeValue
) -> Tensor:
    """Compute StateSpaceModel."""
    return record_op("StateSpaceModel", [x, a, b, c])
