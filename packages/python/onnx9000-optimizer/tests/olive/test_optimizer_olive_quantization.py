"""Tests for Quantization operations."""

from onnx9000.core.ir import DynamicDim, Graph, Node, Tensor
from onnx9000.optimizer.olive.quantization_ops import Quantizer


def test_quantizer_calc_scale() -> None:
    """Test calc_scale_zp."""
    (scale, zp) = Quantizer.calc_scale_zp(-10.0, 10.0, "INT8", symmetric=True)
    assert scale > 0
    assert zp == 0
    (scale, zp) = Quantizer.calc_scale_zp(-10.0, 10.0, "INT8", symmetric=False)
    assert scale > 0
    (scale, zp) = Quantizer.calc_scale_zp(0.0, 0.0, "INT8", symmetric=False)
    assert scale == 1.0


def test_quantizer_node_maps() -> None:
    """Test node mapping functions."""
    g = Graph("test")
    n = Node("Dummy", [], [], {}, "n1")
    Quantizer.map_fp32_to_int8_dyn(g, n)
    assert n.attributes["quantized"] is True
    n2 = Node("Dummy", [], [], {}, "n2")
    Quantizer.map_fp32_to_uint8_dyn(g, n2)
    assert n2.attributes["quantized"] is True
    n3 = Node("Dummy", [], [], {}, "n3")
    Quantizer.map_matmul_to_dyn(g, n3)
    assert n3.attributes["quantized"] is True
    n4 = Node("Dummy", [], [], {}, "n4")
    Quantizer.map_matmul_to_int(g, n4)
    assert n4.attributes["quantized"] is True
    n5 = Node("Dummy", [], [], {}, "n5")
    Quantizer.map_conv_to_qlinear(g, n5)
    assert n5.attributes["quantized"] is True
    n6 = Node("Dummy", [], [], {}, "n6")
    Quantizer.map_add_to_qlinear(g, n6)
    assert n6.attributes["quantized"] is True
    n7 = Node("Dummy", [], [], {}, "n7")
    Quantizer.inject_fake_quantization(g, n7)
    assert n7.attributes["quantized"] is True
    n8 = Node("Dummy", [], [], {}, "n8")
    Quantizer.ensure_bias_precision(n8)
    assert n8.attributes["quantized"] is True
    n9 = Node("NonZero", [], [], {}, "n9")
    assert Quantizer.highlight_non_quantizable(n9) is True


def test_quantizer_graph_maps() -> None:
    """Test graph level modifications."""
    g = Graph("test")
    g.metadata = {}
    Quantizer.fold_qdq(g)
    assert g.metadata["qdq_folded"] is True
    Quantizer.fuse_batchnorm_into_conv(g)
    assert g.metadata["bn_fused"] is True
    Quantizer.quantize_constants_to_initializers(g)
    assert g.metadata["constants_quantized"] is True
    Quantizer.fallback_to_fp32(g, [])
    assert g.metadata["fallback"] is True
    Quantizer.apply_fp16_mixed_precision(g)
    assert g.metadata["fp16"] is True
    Quantizer.inject_int8_fp32_boundaries(g)
    assert g.metadata["boundaries"] is True
    Quantizer.inject_webgpu_shader_unpacking(g)
    assert g.metadata["webgpu"] is True


def test_quantizer_tensor_maps() -> None:
    """Test tensor operations."""
    t = Tensor("t1", "FLOAT32", [DynamicDim("N")], [1.0])
    assert Quantizer.block_wise_quantization(t) is t
    assert Quantizer.k_means_clustering(t) is t
    assert Quantizer.quantize_int4(t) is t
    assert Quantizer.pack_int4_to_int8(t, t) is t
    assert Quantizer.pack_int4_to_uint32_webgpu(t) is t
    Quantizer.per_channel_quant_conv(t)
    assert t.name == "t1_quant"
    t2 = Tensor("t2", "FLOAT32", [DynamicDim("N")], [1.0])
    Quantizer.per_channel_quant_matmul(t2)
    assert t2.name == "t2_quant_mm"
    assert Quantizer.handle_pytorch_qint(t) is t
    assert Quantizer.awq_quantization(t) is t
    assert Quantizer.gptq_quantization(t) is t


def test_quantizer_metrics() -> None:
    """Test metrics tracking."""
    t = Tensor("t1", "FLOAT32", [DynamicDim("N")], [1.0])
    assert Quantizer.extract_bounds(t) == (-1.0, 1.0)
    assert "dims" in Quantizer.track_metrics(t)
    assert Quantizer.calibrate_histogram([]) == (-1.0, 1.0)
    assert Quantizer.calibrate_entropy([]) == (-1.0, 1.0)
    assert Quantizer.calibrate_minmax([]) == (-1.0, 1.0)
    assert Quantizer.verify_opset_limits(10, True) is True
    assert Quantizer.extract_fp16_scale() == 1.0
    assert Quantizer.extract_bf16_scale() == 1.0
    assert Quantizer.calculate_psnr([], []) == 100.0
