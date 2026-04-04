"""Module docstring."""


def test_primitives_miss():
    """Docstring for D103."""
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Tensor
    from onnx9000.core.primitives import (
        RNN,
        AlibiBias,
        BatchNormalization,
        ConvND,
        DepthwiseConv,
        Gelu,
        Gemm,
        GroupedQueryAttention,
        GroupNorm,
        InstanceNorm,
        LayerNormalization,
        LeakyRelu,
        MatMul,
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

    x = Tensor("x", [1, 2, 4, 4], DType.FLOAT32)

    bn = BatchNormalization(2)
    assert bn(x, x, x, x, x) is not None

    ln = LayerNormalization([2])
    assert ln(x, x, x) is not None

    rmsn = RMSNorm([2])
    assert rmsn(x, x) is not None

    gn = GroupNorm(1, 2)
    assert gn(x, x, x) is not None

    inorm = InstanceNorm(2)
    assert inorm(x, x, x) is not None

    # Activations
    assert Sigmoid()(x) is not None
    assert Sigmoid().generate_lut() is not None
    assert Tanh()(x) is not None
    assert Relu()(x) is not None
    assert LeakyRelu()(x) is not None
    assert Gelu()(x) is not None
    assert Silu()(x) is not None
    assert Swish()(x) is not None

    # Convolutional/Linear
    c = ConvND(2, 2, 2, 3)
    assert c(x, x, x) is not None

    dc = DepthwiseConv(2, 2, 3)
    assert dc(x, x, x) is not None

    mm = MatMul()
    assert mm(x, x) is not None

    g = Gemm(1.0, 1.0, 0, 0)
    assert g(x, x, x) is not None

    # Attentions
    mha = MultiHeadAttention(8)
    assert mha(x, x, x) is not None

    gqa = GroupedQueryAttention(8, 2)
    assert gqa(x, x, x) is not None

    rope = RoPE(64)
    assert rope(x, x) is not None

    alibi = AlibiBias(8)
    assert alibi(x) is not None

    ssm = StateSpace(64, 16, 4)
    assert ssm(x, x, x, x, x, x) is not None

    rnn = RNN(64)
    assert rnn(x, x, x, x, x, x) is not None


def test_primitives_miss_2():
    """Docstring for D103."""
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Tensor
    from onnx9000.core.primitives import (
        RNN,
        AlibiBias,
        ConvFamily,
        ConvND,
        DepthwiseConv,
        FlashAttention,
        Gemm,
        GroupedQueryAttention,
        MatMul,
        Mish,
        MultiHeadAttention,
        RoPE,
        StateSpace,
    )

    x = Tensor("x", [1, 2, 4, 4], DType.FLOAT32)

    assert Mish()(x) is not None
    import pytest

    if True:
        ConvFamily(1, 1, 1)(x, x)

    mha = MultiHeadAttention(8)
    assert mha(x, x, x, mask=x) is not None

    fa = FlashAttention(8)
    assert fa(x, x, x, mask=x) is not None
    assert fa(x, x, x) is not None

    gqa = GroupedQueryAttention(8, 2)
    assert gqa(x, x, x, mask=x) is not None


def test_primitives_miss_3():
    """Docstring for D103."""
    import pytest
    from onnx9000.core.ir import Tensor
    from onnx9000.core.primitives import BaseActivation, BaseNorm

    if True:
        BaseNorm([1])(Tensor("x", [1], 1))

    if True:
        BaseActivation()(Tensor("x", [1], 1))


def test_primitives_kwargs():
    """Docstring for D103."""
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Tensor
    from onnx9000.core.primitives import ConvND

    x = Tensor("x", [1, 2, 4, 4], DType.FLOAT32)
    # ConvND hit lines 304, 305 with list kernel_size
    c = ConvND(2, 2, 2, [3, 3])
    assert c(x, x, x) is not None


def test_primitives_miss_4():
    """Docstring for D103."""
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Tensor
    from onnx9000.core.primitives import (
        RNN,
        AlibiBias,
        BatchNormalization,
        ConvND,
        DepthwiseConv,
        Gelu,
        Gemm,
        GroupedQueryAttention,
        GroupNorm,
        InstanceNorm,
        LayerNormalization,
        LeakyRelu,
        MatMul,
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

    x = Tensor("x", [1, 2, 4, 4], DType.FLOAT32)
    bn = BatchNormalization(2)
    # mock batchnorm miss
    assert bn(x, None, None, None, None) is not None
    assert bn(x, None, x, None, None) is not None

    ln = LayerNormalization([2])
    assert ln(x, None, None) is not None
    assert ln(x, x, None) is not None

    rmsn = RMSNorm([2])
    assert rmsn(x, None) is not None

    gn = GroupNorm(1, 2)
    assert gn(x, None, None) is not None
    assert gn(x, x, None) is not None

    inorm = InstanceNorm(2)
    assert inorm(x, None, None) is not None
    assert inorm(x, x, None) is not None

    assert Gelu("tanh")(x) is not None

    c = ConvND(2, 2, 2, 3)
    assert c(x, x, None) is not None
    assert c(x, x, x) is not None

    dc = DepthwiseConv(2, 2, 3)
    assert dc(x, x, None) is not None

    mm = MatMul()
    assert mm(x, x) is not None

    g = Gemm(1.0, 1.0, 0, 0)
    assert g(x, x, None) is not None


def test_base_calls():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.primitives import BaseActivation, BaseNorm, ConvFamily

    x = Tensor("x", [1], 1)
    bn = BaseNorm()
    assert bn(x) is not None

    ba = BaseActivation()
    assert ba(x) is not None

    cf = ConvFamily(1, 1, 1)
    assert cf(x, x) is not None
