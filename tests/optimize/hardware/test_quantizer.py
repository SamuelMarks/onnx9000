"""Test quantizer."""

import pytest
import numpy as np
from onnx9000.optimize.hardware.quantizer import Quantizer


def test_quantizer_init():
    """Provides semantic functionality and verification."""
    q = Quantizer()
    assert q is not None


def test_calculate_scale_zero_point_asymmetric():
    """Provides semantic functionality and verification."""
    scale, zp = Quantizer.calculate_scale_zero_point(
        0.0, 255.0, 0, 255, symmetric=False
    )
    assert np.isclose(scale, 1.0)
    assert zp == 0


def test_calculate_scale_zero_point_symmetric():
    """Provides semantic functionality and verification."""
    scale, zp = Quantizer.calculate_scale_zero_point(
        -127.0, 127.0, -127, 127, symmetric=True
    )
    assert np.isclose(scale, 1.0)
    assert zp == 0


def test_quantize_asymmetric():
    """Provides semantic functionality and verification."""
    tensor = np.array([-10.0, 0.0, 10.0])
    q, s, zp = Quantizer.quantize_asymmetric(tensor)
    assert q.dtype == np.uint8
    assert q.shape == tensor.shape


def test_quantize_symmetric():
    """Provides semantic functionality and verification."""
    tensor = np.array([-10.0, 0.0, 10.0])
    q, s, zp = Quantizer.quantize_symmetric(tensor)
    assert q.dtype == np.int8
    assert q.shape == tensor.shape


def test_dynamic_quantize_linear():
    """Provides semantic functionality and verification."""
    tensor = np.array([-10.0, 0.0, 10.0])
    q, s, zp = Quantizer.dynamic_quantize_linear(tensor)
    assert q.dtype == np.uint8
    assert q.shape == tensor.shape


def test_quantize_linear():
    """Provides semantic functionality and verification."""
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    scale = np.array(0.1)
    zp = np.array(0, dtype=np.uint8)
    q = Quantizer.quantize_linear(tensor, scale, zp, axis=1)
    assert q.dtype == np.uint8


def test_dequantize_linear():
    """Provides semantic functionality and verification."""
    q = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    scale = np.array(0.1)
    zp = np.array(0, dtype=np.uint8)
    tensor = Quantizer.dequantize_linear(q, scale, zp, axis=1)
    assert tensor.dtype == np.float32


def test_fake_quantize():
    """Provides semantic functionality and verification."""
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    scale = np.array(0.1)
    zp = np.array(0, dtype=np.uint8)
    fq = Quantizer.fake_quantize(tensor, scale, zp, axis=1)
    assert fq.shape == tensor.shape


def test_matmul_integer():
    """Provides semantic functionality and verification."""
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    b = np.array([[1, 0], [0, 1]], dtype=np.int32)
    a_zp = np.array(0, dtype=np.int32)
    b_zp = np.array(0, dtype=np.int32)
    out = Quantizer.matmul_integer(a, b, a_zp, b_zp)
    assert out.shape == (2, 2)
    assert np.array_equal(out, a)


def test_conv_integer():
    """Provides semantic functionality and verification."""
    x = np.ones((1, 1, 3, 3), dtype=np.int32)
    w = np.ones((1, 1, 2, 2), dtype=np.int32)
    x_zp = np.array(0, dtype=np.int32)
    w_zp = np.array(0, dtype=np.int32)
    out = Quantizer.conv_integer(x, w, x_zp, w_zp)
    assert out.shape == (1, 1, 2, 2)


def test_qlinear_conv():
    """Provides semantic functionality and verification."""
    x = np.ones((1, 1, 3, 3), dtype=np.uint8)
    w = np.ones((1, 1, 2, 2), dtype=np.uint8)
    scale = np.array(1.0, dtype=np.float32)
    zp = np.array(0, dtype=np.uint8)
    out = Quantizer.qlinear_conv(x, scale, zp, w, scale, zp, scale, zp)
    assert out.dtype == np.uint8


def test_qlinear_matmul():
    """Provides semantic functionality and verification."""
    a = np.ones((2, 2), dtype=np.uint8)
    b = np.ones((2, 2), dtype=np.uint8)
    scale = np.array(1.0, dtype=np.float32)
    zp = np.array(0, dtype=np.uint8)
    out = Quantizer.qlinear_matmul(a, scale, zp, b, scale, zp, scale, zp)
    assert out.dtype == np.uint8


def test_qlinear_add():
    """Provides semantic functionality and verification."""
    a = np.ones((2, 2), dtype=np.uint8)
    b = np.ones((2, 2), dtype=np.uint8)
    scale = np.array(1.0, dtype=np.float32)
    zp = np.array(0, dtype=np.uint8)
    out = Quantizer.qlinear_add(a, scale, zp, b, scale, zp, scale, zp)
    assert out.dtype == np.uint8


def test_qlinear_sigmoid():
    """Provides semantic functionality and verification."""
    x = np.ones((2, 2), dtype=np.uint8)
    scale = np.array(1.0, dtype=np.float32)
    zp = np.array(0, dtype=np.uint8)
    out = Quantizer.qlinear_sigmoid(x, scale, zp, scale, zp)
    assert out.dtype == np.uint8


def test_qlinear_leakyrelu():
    """Provides semantic functionality and verification."""
    x = np.ones((2, 2), dtype=np.uint8)
    scale = np.array(1.0, dtype=np.float32)
    zp = np.array(0, dtype=np.uint8)
    out = Quantizer.qlinear_leakyrelu(x, scale, zp, scale, zp)
    assert out.dtype == np.uint8


def test_activation_clipping():
    """Provides semantic functionality and verification."""
    tensor = np.array([-10.0, 0.0, 10.0])
    clipped = Quantizer.activation_clipping(tensor, 0.0, 6.0)
    assert np.array_equal(clipped, np.array([0.0, 0.0, 6.0]))


def test_per_tensor_quantization():
    """Provides semantic functionality and verification."""
    tensor = np.array([-10.0, 0.0, 10.0])
    q, s, zp = Quantizer.per_tensor_quantization(tensor)
    assert q.shape == tensor.shape


def test_per_channel_quantization():
    """Provides semantic functionality and verification."""
    tensor = np.array([[-10.0, 10.0], [-5.0, 5.0]])
    q, s, zp = Quantizer.per_channel_quantization(tensor, axis=0)
    assert q.shape == tensor.shape
    assert s.shape == (2,)


def test_cross_entropy_calibration():
    """Provides semantic functionality and verification."""
    data = [np.array([-10.0, 0.0, 10.0])]
    min_t, max_t = Quantizer.cross_entropy_calibration(data, num_bins=256)
    assert min_t <= 0
    assert max_t >= 0


def test_kl_divergence_calibration_zero():
    """Provides semantic functionality and verification."""
    data = [np.zeros(10)]
    min_t, max_t = Quantizer.kl_divergence_calibration(data)
    assert min_t == 0.0
    assert max_t == 0.0


def test_percentile_calibration():
    """Provides semantic functionality and verification."""
    data = [np.arange(100).astype(np.float32)]
    min_t, max_t = Quantizer.percentile_calibration(data, percentile=90)
    assert np.isclose(max_t, 89.1)


def test_moving_average_calibration():
    """Provides semantic functionality and verification."""
    data_stream = [np.array([-1.0, 1.0]), np.array([-2.0, 2.0])]
    min_t, max_t = Quantizer.moving_average_calibration(data_stream, momentum=0.9)
    assert min_t < 0
    assert max_t > 0


def test_quantization_error():
    """Provides semantic functionality and verification."""
    orig = np.array([0.0, 1.0, 2.0])
    q = np.array([0, 10, 20], dtype=np.uint8)
    scale = np.array(0.1)
    zp = np.array(0, dtype=np.uint8)
    error = Quantizer.quantization_error(orig, q, scale, zp, axis=0)
    assert isinstance(error, float)


def test_skip_layer_heuristic():
    """Provides semantic functionality and verification."""
    assert Quantizer.skip_layer_heuristic(0.05, 0.01) is True
    assert Quantizer.skip_layer_heuristic(0.001, 0.01) is False


def test_int32_accumulation_scaling():
    """Provides semantic functionality and verification."""
    acc = np.array([100], dtype=np.int32)
    a_scale = np.array(0.1)
    b_scale = np.array(0.2)
    out_scale = np.array(0.05)
    scaled = Quantizer.int32_accumulation_scaling(acc, a_scale, b_scale, out_scale)
    assert scaled.shape == acc.shape


def test_dynamic_quantize_matmul():
    """Provides semantic functionality and verification."""
    a = np.array([[1.0, 2.0]])
    b = np.array([[3.0], [4.0]])
    out = Quantizer.dynamic_quantize_matmul(a, b)
    assert out.shape == (1, 1)


def test_quantize_fp16():
    """Provides semantic functionality and verification."""
    tensor = np.array([1.0, 2.0], dtype=np.float32)
    out = Quantizer.quantize_fp16(tensor)
    assert out.dtype == np.float16


def test_quantize_bf16():
    """Provides semantic functionality and verification."""
    tensor = np.array([1.0, 2.0], dtype=np.float32)
    out = Quantizer.quantize_bf16(tensor)
    assert out.dtype == np.float32


class MockTensorProto:
    """Represents the MockTensorProto class."""

    def __init__(self, dt, rd=None, fd=None):
        """Provides   init   functionality and verification."""
        self.data_type = dt
        self.raw_data = rd
        self.float_data = fd


def test_convert_initializer_to_int8_none():
    """Provides semantic functionality and verification."""
    assert Quantizer.convert_initializer_to_int8(None) is None


def test_convert_initializer_to_int8_raw():
    """Provides semantic functionality and verification."""
    import numpy as np

    tp = MockTensorProto(1, rd=np.array([1.0, -1.0], dtype=np.float32).tobytes())
    res = Quantizer.convert_initializer_to_int8(tp)
    assert res.data_type == 3
    assert res.raw_data is not None


def test_convert_initializer_to_int8_float_data():
    """Provides semantic functionality and verification."""
    tp = MockTensorProto(1, fd=[1.0, -1.0])
    res = Quantizer.convert_initializer_to_int8(tp)
    assert res.data_type == 3


def test_convert_initializer_to_int8_already_int8():
    """Provides semantic functionality and verification."""
    tp = MockTensorProto(3)
    res = Quantizer.convert_initializer_to_int8(tp)
    assert res.data_type == 3


def test_convert_initializer_to_int8_no_data():
    """Provides semantic functionality and verification."""
    tp = MockTensorProto(1)
    res = Quantizer.convert_initializer_to_int8(tp)
    assert res.data_type == 1


def test_dynamic_dispatcher():
    """Provides semantic functionality and verification."""
    assert Quantizer.dynamic_dispatcher(np.array([1.0, 2.0])) == "asymmetric"
    assert Quantizer.dynamic_dispatcher(np.array([-2.0, -1.0])) == "asymmetric"
    assert Quantizer.dynamic_dispatcher(np.array([-2.0, 2.0])) == "symmetric"
    assert Quantizer.dynamic_dispatcher(np.array([-10.0, 1.0])) == "asymmetric"


def test_quantize_int4_asymmetric():
    """Provides semantic functionality and verification."""
    tensor = np.array([-10.0, 0.0, 10.0])
    q, s, zp = Quantizer.quantize_int4_asymmetric(tensor)
    assert q.dtype == np.uint8


def test_quantize_int4_symmetric():
    """Provides semantic functionality and verification."""
    tensor = np.array([-10.0, 0.0, 10.0])
    q, s, zp = Quantizer.quantize_int4_symmetric(tensor)
    assert q.dtype == np.int8


def test_pack_int4_little_endian():
    """Provides semantic functionality and verification."""
    tensor = np.array([1, 2, 3, 4], dtype=np.uint8)
    packed = Quantizer.pack_int4(tensor, little_endian=True)
    assert packed[0] == 2 << 4 | 1
    assert packed[1] == 4 << 4 | 3


def test_pack_int4_big_endian():
    """Provides semantic functionality and verification."""
    tensor = np.array([1, 2], dtype=np.uint8)
    packed = Quantizer.pack_int4(tensor, little_endian=False)
    assert packed[0] == 1 << 4 | 2


def test_pack_int4_odd_length():
    """Provides semantic functionality and verification."""
    tensor = np.array([1, 2, 3], dtype=np.uint8)
    packed = Quantizer.pack_int4(tensor, little_endian=True)
    assert len(packed) == 2


def test_unpack_int4_little_endian():
    """Provides semantic functionality and verification."""
    packed = np.array([2 << 4 | 1, 4 << 4 | 3], dtype=np.uint8)
    unpacked = Quantizer.unpack_int4(packed, length=4, little_endian=True)
    assert np.array_equal(unpacked, np.array([1, 2, 3, 4]))


def test_unpack_int4_big_endian():
    """Provides semantic functionality and verification."""
    packed = np.array([1 << 4 | 2, 3 << 4 | 4], dtype=np.uint8)
    unpacked = Quantizer.unpack_int4(packed, length=4, little_endian=False)
    assert np.array_equal(unpacked, np.array([1, 2, 3, 4]))


def test_block_quantize_linear():
    """Provides semantic functionality and verification."""
    tensor = np.arange(64, dtype=np.float32).reshape(2, 32)
    q, s, zp = Quantizer.block_quantize_linear(tensor, block_size=32)
    assert q.shape == (2, 32)
    assert s.shape == (2,)
    assert zp.shape == (2,)


def test_block_quantize_linear_pad():
    """Provides semantic functionality and verification."""
    tensor = np.arange(10, dtype=np.float32)
    q, s, zp = Quantizer.block_quantize_linear(tensor, block_size=32)
    assert q.shape == (10,)


def test_matmul_nbits():
    """Provides semantic functionality and verification."""
    a = np.ones((2, 4), dtype=np.float32)
    b_packed = np.array([17, 17], dtype=np.uint8)
    scales = np.ones(1)
    zps = np.zeros(1)
    out = Quantizer.matmul_nbits(a, b_packed, scales, zps)
    assert out.shape == (2,)


def test_awq_calibration():
    """Provides semantic functionality and verification."""
    w = np.ones((2, 2))
    a = np.ones((2, 2)) * 2
    calibrated = Quantizer.awq_calibration(w, a)
    assert np.array_equal(calibrated, np.array([[2.0, 2.0], [2.0, 2.0]]))


def test_gptq_calibration():
    """Provides semantic functionality and verification."""
    w = np.ones((2, 2))
    h = np.ones((2, 2))
    calib = Quantizer.gptq_calibration(w, h)
    assert calib.shape == w.shape


def test_smooth_quant():
    """Provides semantic functionality and verification."""
    w = np.ones((2, 2))
    a = np.ones((2, 2)) * 2
    w_s, a_s, s = Quantizer.smooth_quant(w, a)
    assert w_s.shape == w.shape
    assert a_s.shape == a.shape


def test_analyze_sparsity():
    """Provides semantic functionality and verification."""
    tensor = np.array([0.0, 1.0, 0.0, 2.0])
    sparsity = Quantizer.analyze_sparsity(tensor, 1e-05)
    assert sparsity == 0.5


def test_pack_sparse_2_4():
    """Provides semantic functionality and verification."""
    tensor = np.array([0.1, 1.0, 0.2, 2.0])
    packed, meta = Quantizer.pack_sparse_2_4(tensor)
    assert packed.shape == (1, 2)
    assert meta.shape == (1, 2)
    assert np.array_equal(packed[0], [1.0, 2.0])


def test_pack_sparse_2_4_pad():
    """Provides semantic functionality and verification."""
    tensor = np.array([0.1, 1.0])
    packed, meta = Quantizer.pack_sparse_2_4(tensor)
    assert packed.shape == (1, 2)


def test_quantize_linear_vector():
    """Provides semantic functionality and verification."""
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    scale = np.array([0.1, 0.2])
    zp = np.array([0, 0], dtype=np.int8)
    q = Quantizer.quantize_linear(tensor, scale, zp, axis=1)
    assert q.dtype == np.int8


def test_dequantize_linear_vector():
    """Provides semantic functionality and verification."""
    q = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    scale = np.array([0.1, 0.2])
    zp = np.array([0, 0], dtype=np.uint8)
    tensor = Quantizer.dequantize_linear(q, scale, zp, axis=1)
    assert tensor.dtype == np.float32


def test_qlinear_conv_bias():
    """Provides semantic functionality and verification."""
    x = np.ones((1, 1, 3, 3), dtype=np.uint8)
    w = np.ones((1, 1, 2, 2), dtype=np.uint8)
    scale = np.array(1.0, dtype=np.float32)
    zp = np.array(0, dtype=np.uint8)
    b = np.array([1], dtype=np.int32)
    out = Quantizer.qlinear_conv(x, scale, zp, w, scale, zp, scale, zp, B=b)
    assert out.dtype == np.uint8
