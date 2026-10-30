"""Unit tests for ONNX9000 core primitives in `onnx9000.core.primitives`."""

from onnx9000.core.ir import Tensor
from onnx9000.core.primitives import (
    RNN,
    AlibiBias,
    BaseActivation,
    BaseNorm,
    BatchNormalization,
    ConvFamily,
    ConvND,
    DepthwiseConv,
    FlashAttention,
    Gelu,
    Gemm,
    GroupedQueryAttention,
    GroupNorm,
    InstanceNorm,
    LayerNormalization,
    LeakyRelu,
    MatMul,
    Mish,
    MultiHeadAttention,
    Relu,
    RMSNorm,
    RoPE,
    Sigmoid,
    Silu,
    StateSpace,
    Swish,
    Tanh,
)


def test_normalization_primitives() -> None:
    """Verify BaseNorm, BatchNormalization, LayerNormalization, RMSNorm, GroupNorm, and InstanceNorm."""
    x = Tensor("x", (1, 8, 4, 4), 1)
    scale = Tensor("scale", (8,), 1)
    bias = Tensor("bias", (8,), 1)
    mean = Tensor("mean", (8,), 1)
    var = Tensor("var", (8,), 1)

    bn_base = BaseNorm(epsilon=1e-5)
    assert bn_base(x) is x

    bn = BatchNormalization(num_features=8)
    assert isinstance(bn(x, scale, bias, mean, var), Tensor)

    ln = LayerNormalization(normalized_shape=(8,))
    assert isinstance(ln(x, scale, bias), Tensor)

    rms = RMSNorm(normalized_shape=(8,))
    assert isinstance(rms(x, scale), Tensor)

    gn = GroupNorm(num_groups=2, num_channels=8)
    assert isinstance(gn(x, scale, bias), Tensor)

    inst_norm = InstanceNorm(num_features=8)
    assert isinstance(inst_norm(x, scale, bias), Tensor)


def test_activation_primitives() -> None:
    """Verify BaseActivation, Relu, Sigmoid, Tanh, LeakyRelu, Gelu, Silu, Swish, and Mish."""
    x = Tensor("x", (1, 8), 1)

    act_base = BaseActivation()
    assert act_base(x) is x
    assert isinstance(act_base.generate_lut(), Tensor)

    for act in [Relu(), Sigmoid(), Tanh(), LeakyRelu(0.1), Gelu(), Silu(), Swish(), Mish()]:
        assert isinstance(act(x), Tensor)


def test_conv_linear_attention_and_recurrent_primitives() -> None:
    """Verify ConvFamily, ConvND, DepthwiseConv, MatMul, Gemm, Attention, RoPE, AlibiBias, StateSpace, and RNN."""
    x = Tensor("x", (1, 8, 8, 8), 1)
    w = Tensor("w", (8, 8, 3, 3), 1)
    b = Tensor("b", (8,), 1)

    cf = ConvFamily(8, 8, 3)
    assert cf(x, w, b) is x

    conv2d = ConvND(2, 8, 8, 3, stride=1, padding=1)
    assert isinstance(conv2d(x, w, b), Tensor)

    dw = DepthwiseConv(2, 8, 3, padding=1)
    assert isinstance(dw(x, w, b), Tensor)

    mm = MatMul()
    assert isinstance(mm(x, x), Tensor)

    gemm = Gemm(trans_b=1)
    assert isinstance(gemm(x, x, b), Tensor)

    mha = MultiHeadAttention(num_heads=2)
    assert isinstance(mha(x, x, x, mask=x), Tensor)

    fa = FlashAttention(num_heads=2)
    assert isinstance(fa(x, x, x, mask=x), Tensor)

    gqa = GroupedQueryAttention(num_heads=4, num_kv_heads=2)
    assert isinstance(gqa(x, x, x, mask=x), Tensor)

    rope = RoPE(dim=8)
    assert isinstance(rope(x, b), Tensor)

    alibi = AlibiBias(num_heads=4)
    assert isinstance(alibi(x), Tensor)

    ssm = StateSpace(d_model=8, d_state=4, d_conv=3)
    assert isinstance(ssm(x, x, x, x, x, x), Tensor)

    rnn = RNN(hidden_size=8)
    assert isinstance(rnn(x, w, w), Tensor)
