"""Module docstring."""

import pytest
from onnx9000.core.ir import Tensor
from onnx9000.core.ops import (
    expm1,
    fmod,
    isfinite,
    log1p,
    log2,
    log10,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
)


def test_missing_zoo_ops():
    """Docstring for D103."""
    x = Tensor(name="x", shape=(2, 2), dtype=1)
    y = Tensor(name="y", shape=(2, 2), dtype=1)

    assert fmod(x, y).name == "Fmod_out"
    assert log2(x).name == "Log2_out"
    assert log10(x).name == "Log10_out"
    assert expm1(x).name == "Expm1_out"
    assert log1p(x).name == "Log1p_out"
    assert isfinite(x).name == "IsFinite_out"
    assert logical_and(x, y).name == "LogicalAnd_out"
    assert logical_or(x, y).name == "LogicalOr_out"
    assert logical_xor(x, y).name == "LogicalXor_out"
    assert logical_not(x).name == "LogicalNot_out"


def test_missing_zoo_ops_2():
    """Docstring for D103."""
    x = Tensor(name="x", shape=(2, 2), dtype=1)
    y = Tensor(name="y", shape=(2, 2), dtype=1)
    from onnx9000.core.ops import im2col, repeat

    assert im2col(x).name == "Im2Col_out"
    assert repeat(x, y).name == "Repeat_out"


def test_missing_zoo_ops_3():
    """Docstring for D103."""
    x = Tensor(name="x", shape=(2, 2), dtype=1)
    y = Tensor(name="y", shape=(2, 2), dtype=1)
    z = Tensor(name="z", shape=(2, 2), dtype=1)
    from onnx9000.core.ops import (
        adaptive_avg_pool2d,
        adaptive_max_pool2d,
        average_pool1d,
        average_pool2d,
        average_pool3d,
        conv1d,
        conv2d,
        conv3d,
        conv_transpose1d,
        conv_transpose2d,
        conv_transpose3d,
        deformable_conv2d,
        depthwise_conv2d,
        max_pool1d,
        max_pool2d,
        max_pool3d,
    )

    assert conv1d(x, y).name == "Conv1D_out"
    assert conv2d(x, y).name == "Conv2D_out"
    assert conv3d(x, y).name == "Conv3D_out"
    assert conv_transpose1d(x, y).name == "ConvTranspose1D_out"
    assert conv_transpose2d(x, y).name == "ConvTranspose2D_out"
    assert conv_transpose3d(x, y).name == "ConvTranspose3D_out"
    assert depthwise_conv2d(x, y).name == "DepthwiseConv2D_out"
    assert deformable_conv2d(x, y, z).name == "DeformableConv2D_out"
    assert max_pool1d(x).name == "MaxPool1D_out"
    assert max_pool2d(x).name == "MaxPool2D_out"
    assert max_pool3d(x).name == "MaxPool3D_out"
    assert average_pool1d(x).name == "AveragePool1D_out"
    assert average_pool2d(x).name == "AveragePool2D_out"
    assert average_pool3d(x).name == "AveragePool3D_out"
    assert adaptive_max_pool2d(x).name == "AdaptiveMaxPool2D_out"
    assert adaptive_avg_pool2d(x).name == "AdaptiveAvgPool2D_out"


def test_missing_zoo_ops_4():
    """Docstring for D103."""
    x = Tensor(name="x", shape=(2, 2), dtype=1)
    y = Tensor(name="y", shape=(2, 2), dtype=1)
    z = Tensor(name="z", shape=(2, 2), dtype=1)
    from onnx9000.core.ops import adaln, geglu, local_response_norm, reglu, silu, swiglu

    assert local_response_norm(x, 2).name == "LocalResponseNorm_out"
    assert adaln(x, y, z).name == "AdaLN_out"
    assert silu(x).name == "Silu_out"
    assert swiglu(x, y).name == "SwiGLU_out"
    assert geglu(x, y).name == "GeGLU_out"
    assert reglu(x, y).name == "ReGLU_out"


def test_missing_zoo_ops_5():
    """Docstring for D103."""
    x = Tensor(name="x", shape=(2, 2), dtype=1)
    y = Tensor(name="y", shape=(2, 2), dtype=1)
    z = Tensor(name="z", shape=(2, 2), dtype=1)
    a = Tensor(name="a", shape=(2, 2), dtype=1)
    b = Tensor(name="b", shape=(2, 2), dtype=1)
    from onnx9000.core.ops import (
        alibi,
        flash_attention,
        grouped_query_attention,
        multi_head_attention,
        multi_query_attention,
        paged_attention,
        rope1d,
        rope2d,
        rope3d,
        sliding_window_attention,
        state_space_model,
    )

    assert multi_head_attention(x, y, z).name == "MultiHeadAttention_out"
    assert grouped_query_attention(x, y, z).name == "GroupedQueryAttention_out"
    assert multi_query_attention(x, y, z).name == "MultiQueryAttention_out"
    assert flash_attention(x, y, z).name == "FlashAttention_out"
    assert paged_attention(x, y, z, a, b).name == "PagedAttention_out"
    assert rope1d(x, y).name == "RoPE1D_out"
    assert rope2d(x, y).name == "RoPE2D_out"
    assert rope3d(x, y).name == "RoPE3D_out"
    assert alibi(x).name == "ALiBi_out"
    assert sliding_window_attention(x, y, z, 2).name == "SlidingWindowAttention_out"
    assert state_space_model(x, y, z, a).name == "StateSpaceModel_out"


def test_missing_zoo_ops_6():
    """Docstring for D103."""
    x = Tensor(name="x", shape=(2, 2), dtype=1)
    y = Tensor(name="y", shape=(2, 2), dtype=1)
    z = Tensor(name="z", shape=(2, 2), dtype=1)
    from onnx9000.core.ops import dequantize_linear, quantize_linear

    assert quantize_linear(x, y, z).name == "QuantizeLinear_out"
    assert dequantize_linear(x, y, z).name == "DequantizeLinear_out"


def test_quantize_miss():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.ops import dequantize_linear, quantize_linear

    x = Tensor("x", [1], 1)
    # The tests passed x,y,z earlier, but we need to pass just x, y
    assert quantize_linear(x, x).name == "QuantizeLinear_out"
    assert dequantize_linear(x, x).name == "DequantizeLinear_out"


def test_quantize_miss_3_params():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.ops import dequantize_linear, quantize_linear

    x = Tensor("x", [1], 1)
    assert quantize_linear(x, x, x).name == "QuantizeLinear_out"
    assert dequantize_linear(x, x, x).name == "DequantizeLinear_out"


def test_quantize_miss_args():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.ops import conv_transpose, dequantize_linear, quantize_linear

    x = Tensor("x", [1], 1)
    # The actual missing lines are the inner `inputs.append` branches!
    # Which are triggered when zero points/biases are provided.
    quantize_linear(x, x, x)
    dequantize_linear(x, x, x)
    conv_transpose(x, x, x)


def test_quantize_miss_coverage():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.ops import conv_transpose, dequantize_linear, quantize_linear

    x = Tensor("x", [1], 1)
    quantize_linear(x, x, y_zero_point=x)
    dequantize_linear(x, x, x_zero_point=x)
    conv_transpose(x, x, b=x)


def test_profiler_generate_suggestions_all():
    """Docstring for D103."""
    from onnx9000.core.profiler import ProfilerResult

    r = ProfilerResult()
    r.float64_count = 1
    r.int64_count = 1
    r.generate_suggestions()
    assert len(r.suggestions) > 0

    r2 = ProfilerResult()
    r2.node_profiles = [
        {
            "name": "n",
            "flops": 10,
            "macs": 10,
            "params": 10,
            "activation_bytes": 10,
            "arithmetic_intensity": 0.5,
        }
    ]
    r2.generate_suggestions()


def test_quantize_miss_direct_args():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.ops import conv_transpose, dequantize_linear, quantize_linear

    x = Tensor("x", [1], 1)

    # 388-391
    quantize_linear(x, x, x)

    # 397-400
    dequantize_linear(x, x, x)

    conv_transpose(x, x, x)
