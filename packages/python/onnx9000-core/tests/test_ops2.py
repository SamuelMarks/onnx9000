"""Test missing ops."""

from onnx9000.core.ir import Tensor
from onnx9000.core.ops import (
    batch_normalization,
    bit_cast,
    cum_prod,
    flatten,
    gelu,
    global_lp_pool,
    grid_sample,
    hamming_window,
    hann_window,
    identity,
    if_op,
    image_decoder,
    log,
    loop,
    mat_mul_integer,
    max_roi_pool,
    max_unpool,
    negative_log_likelihood_loss,
    optional,
    optional_get_element,
    optional_has_element,
    pad,
    q_linear_conv,
    q_linear_mat_mul,
    rms_normalization,
    range_op2,
    roi_align,
    rotary_embedding,
    stft,
    scan,
    shape,
    softmax_cross_entropy_loss,
    split,
    sqrt,
    squeeze,
    tensor_scatter,
    tf_idf_vectorizer,
    unsqueeze,
    upsample,
    xor,
)


def test_batch_normalization() -> None:
    """Test batch_normalization."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = batch_normalization(t)
    assert res.name == "BatchNormalization_out"


def test_bit_cast() -> None:
    """Test bit_cast."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = bit_cast(t)
    assert res.name == "BitCast_out"


def test_cum_prod() -> None:
    """Test cum_prod."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = cum_prod(t)
    assert res.name == "CumProd_out"


def test_flatten() -> None:
    """Test flatten."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = flatten(t)
    assert res.name == "Flatten_out"


def test_gelu() -> None:
    """Test gelu."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = gelu(t)
    assert res.name == "Gelu_out"


def test_global_lp_pool() -> None:
    """Test global_lp_pool."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = global_lp_pool(t)
    assert res.name == "GlobalLpPool_out"


def test_grid_sample() -> None:
    """Test grid_sample."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = grid_sample(t)
    assert res.name == "GridSample_out"


def test_hamming_window() -> None:
    """Test hamming_window."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = hamming_window(t)
    assert res.name == "HammingWindow_out"


def test_hann_window() -> None:
    """Test hann_window."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = hann_window(t)
    assert res.name == "HannWindow_out"


def test_identity() -> None:
    """Test identity."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = identity(t)
    assert res.name == "Identity_out"


def test_if_op() -> None:
    """Test if_op."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = if_op(t)
    assert res.name == "If_out"


def test_image_decoder() -> None:
    """Test image_decoder."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = image_decoder(t)
    assert res.name == "ImageDecoder_out"


def test_log() -> None:
    """Test log."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = log(t)
    assert res.name == "Log_out"


def test_loop() -> None:
    """Test loop."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = loop(t)
    assert res.name == "Loop_out"


def test_mat_mul_integer() -> None:
    """Test mat_mul_integer."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = mat_mul_integer(t)
    assert res.name == "MatMulInteger_out"


def test_max_roi_pool() -> None:
    """Test max_roi_pool."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = max_roi_pool(t)
    assert res.name == "MaxRoiPool_out"


def test_max_unpool() -> None:
    """Test max_unpool."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = max_unpool(t)
    assert res.name == "MaxUnpool_out"


def test_negative_log_likelihood_loss() -> None:
    """Test negative_log_likelihood_loss."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = negative_log_likelihood_loss(t)
    assert res.name == "NegativeLogLikelihoodLoss_out"


def test_optional() -> None:
    """Test optional."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = optional(t)
    assert res.name == "Optional_out"


def test_optional_get_element() -> None:
    """Test optional_get_element."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = optional_get_element(t)
    assert res.name == "OptionalGetElement_out"


def test_optional_has_element() -> None:
    """Test optional_has_element."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = optional_has_element(t)
    assert res.name == "OptionalHasElement_out"


def test_pad() -> None:
    """Test pad."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = pad(t)
    assert res.name == "Pad_out"


def test_q_linear_conv() -> None:
    """Test q_linear_conv."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = q_linear_conv(t)
    assert res.name == "QLinearConv_out"


def test_q_linear_mat_mul() -> None:
    """Test q_linear_mat_mul."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = q_linear_mat_mul(t)
    assert res.name == "QLinearMatMul_out"


def test_rms_normalization() -> None:
    """Test rms_normalization."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = rms_normalization(t)
    assert res.name == "RMSNormalization_out"


def test_range_op2() -> None:
    """Test range_op2."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = range_op2(t)
    assert res.name == "Range_out"


def test_roi_align() -> None:
    """Test roi_align."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = roi_align(t)
    assert res.name == "RoiAlign_out"


def test_rotary_embedding() -> None:
    """Test rotary_embedding."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = rotary_embedding(t)
    assert res.name == "RotaryEmbedding_out"


def test_stft() -> None:
    """Test stft."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = stft(t)
    assert res.name == "STFT_out"


def test_scan() -> None:
    """Test scan."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = scan(t)
    assert res.name == "Scan_out"


def test_shape() -> None:
    """Test shape."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = shape(t)
    assert res.name == "Shape_out"


def test_softmax_cross_entropy_loss() -> None:
    """Test softmax_cross_entropy_loss."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = softmax_cross_entropy_loss(t)
    assert res.name == "SoftmaxCrossEntropyLoss_out"


def test_split() -> None:
    """Test split."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = split(t)
    assert res.name == "Split_out"


def test_sqrt() -> None:
    """Test sqrt."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = sqrt(t)
    assert res.name == "Sqrt_out"


def test_squeeze() -> None:
    """Test squeeze."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = squeeze(t)
    assert res.name == "Squeeze_out"


def test_tensor_scatter() -> None:
    """Test tensor_scatter."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = tensor_scatter(t)
    assert res.name == "TensorScatter_out"


def test_tf_idf_vectorizer() -> None:
    """Test tf_idf_vectorizer."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = tf_idf_vectorizer(t)
    assert res.name == "TfIdfVectorizer_out"


def test_unsqueeze() -> None:
    """Test unsqueeze."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = unsqueeze(t)
    assert res.name == "Unsqueeze_out"


def test_upsample() -> None:
    """Test upsample."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = upsample(t)
    assert res.name == "Upsample_out"


def test_xor() -> None:
    """Test xor."""
    t = Tensor(name="x", shape=(1,), dtype=1)
    res = xor(t)
    assert res.name == "Xor_out"
