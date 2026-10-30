"""Tests for the primitives module."""

from onnx9000.core.ir import Tensor
from onnx9000.core.primitives import (
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
    Swish,
    Tanh,
)


def test_base_norm_abstract() -> None:
    """Test that BaseNorm is abstract."""
    norm = BaseNorm()
    assert norm(Tensor(name="x", shape=[1], dtype=1)) is not None


def test_batch_normalization() -> None:
    """Test BatchNormalization primitive."""
    norm = BatchNormalization(num_features=3)
    x = Tensor(name="x", shape=[1, 3, 2, 2], dtype=1)
    scale = Tensor(name="scale", shape=[3], dtype=1)
    b = Tensor(name="b", shape=[3], dtype=1)
    mean = Tensor(name="mean", shape=[3], dtype=1)
    var = Tensor(name="var", shape=[3], dtype=1)
    out = norm(x, scale, b, mean, var)
    assert out.name == "BatchNormalization_out"


def test_layer_normalization() -> None:
    """Test LayerNormalization primitive."""
    norm = LayerNormalization(normalized_shape=[2, 2])
    x = Tensor(name="x", shape=[1, 3, 2, 2], dtype=1)
    scale = Tensor(name="scale", shape=[2, 2], dtype=1)
    b = Tensor(name="b", shape=[2, 2], dtype=1)
    out = norm(x, scale, b)
    assert out.name == "LayerNormalization_out"


def test_rms_norm() -> None:
    """Test RMSNorm primitive."""
    norm = RMSNorm(normalized_shape=[2, 2])
    x = Tensor(name="x", shape=[1, 3, 2, 2], dtype=1)
    scale = Tensor(name="scale", shape=[2, 2], dtype=1)
    out = norm(x, scale)
    assert out.name == "RMSNormalization_out"


def test_group_norm() -> None:
    """Test GroupNorm primitive."""
    norm = GroupNorm(num_groups=2, num_channels=4)
    x = Tensor(name="x", shape=[1, 4, 2, 2], dtype=1)
    scale = Tensor(name="scale", shape=[4], dtype=1)
    b = Tensor(name="b", shape=[4], dtype=1)
    out = norm(x, scale, b)
    assert out.name == "GroupNormalization_out"


def test_instance_norm() -> None:
    """Test InstanceNorm primitive."""
    norm = InstanceNorm(num_features=4)
    x = Tensor(name="x", shape=[1, 4, 2, 2], dtype=1)
    scale = Tensor(name="scale", shape=[4], dtype=1)
    b = Tensor(name="b", shape=[4], dtype=1)
    out = norm(x, scale, b)
    assert out.name == "InstanceNormalization_out"


def test_base_activation() -> None:
    """Verify that base activation behaves as expected."""
    act = BaseActivation()
    assert act(Tensor(name="x", shape=[1], dtype=1)) is not None
    lut = act.generate_lut()
    assert lut.name == "Constant_out"


def test_activations() -> None:
    """Verify that activations behaves as expected."""
    x = Tensor(name="x", shape=[1], dtype=1)
    assert Relu()(x).name == "Relu_out"
    assert Sigmoid()(x).name == "Sigmoid_out"
    assert Tanh()(x).name == "Tanh_out"
    assert LeakyRelu(0.01)(x).name == "LeakyRelu_out"
    assert Gelu()(x).name == "Gelu_out"
    assert Silu()(x).name == "Swish_out"
    assert Swish()(x).name == "Swish_out"
    assert Mish()(x).name == "Mish_out"


def test_conv_family() -> None:
    """Verify that conv family behaves as expected."""
    fam = ConvFamily(1, 1, 3)
    assert (
        fam(Tensor(name="x", shape=[1], dtype=1), Tensor(name="w", shape=[1], dtype=1)) is not None
    )


def test_conv_nd() -> None:
    """Verify that conv nd behaves as expected."""
    conv = ConvND(2, 3, 4, 3)
    x = Tensor(name="x", shape=[1, 3, 8, 8], dtype=1)
    w = Tensor(name="w", shape=[4, 3, 3, 3], dtype=1)
    out = conv(x, w)
    assert out.name == "Conv_out"


def test_depthwise_conv() -> None:
    """Verify that depthwise conv behaves as expected."""
    conv = DepthwiseConv(2, 3, 3)
    x = Tensor(name="x", shape=[1, 3, 8, 8], dtype=1)
    w = Tensor(name="w", shape=[3, 1, 3, 3], dtype=1)
    out = conv(x, w)
    assert out.name == "Conv_out"
    assert conv.groups == 3


def test_matmul_gemm() -> None:
    """Verify that matmul gemm behaves as expected."""
    x = Tensor(name="x", shape=[2, 2], dtype=1)
    y = Tensor(name="y", shape=[2, 2], dtype=1)
    assert MatMul()(x, y).name == "MatMul_out"
    assert Gemm()(x, y).name == "Gemm_out"


def test_attention() -> None:
    """Verify that attention behaves as expected."""
    q = Tensor(name="q", shape=[1, 8, 32, 64], dtype=1)
    k = Tensor(name="k", shape=[1, 8, 32, 64], dtype=1)
    v = Tensor(name="v", shape=[1, 8, 32, 64], dtype=1)

    mha = MultiHeadAttention(8)
    assert mha(q, k, v).name == "Attention_out"

    fa = FlashAttention(8)
    assert fa(q, k, v).name == "FlashAttention_out"

    gqa = GroupedQueryAttention(8, 2)
    assert gqa(q, k, v).name == "GroupedQueryAttention_out"


def test_rope_alibi() -> None:
    """Verify that rope alibi behaves as expected."""
    x = Tensor(name="x", shape=[1, 32, 64], dtype=1)
    pos = Tensor(name="pos", shape=[1, 32], dtype=1)
    assert RoPE(64)(x, pos).name == "RoPE_out"

    mask = Tensor(name="mask", shape=[1, 8, 32, 32], dtype=1)
    assert AlibiBias(8)(mask).name == "AlibiBias_out"
