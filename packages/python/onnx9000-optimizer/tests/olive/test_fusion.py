"""Tests for Graph Fusions & Pattern Optimization."""

from onnx9000.core.ir import Graph
from onnx9000.optimizer.olive.fusion_ops import FusionOptimizer


def test_fusion_ops() -> None:
    """Test all fusion optimizations."""
    g = Graph("test")
    g.metadata = {}
    FusionOptimizer.fuse_conv_relu(g)
    assert g.metadata["fused_conv_relu"] is True
    FusionOptimizer.fuse_conv_clip(g)
    assert g.metadata["fused_conv_clip"] is True
    FusionOptimizer.fuse_conv_sigmoid(g)
    assert g.metadata["fused_conv_sigmoid"] is True
    FusionOptimizer.fuse_matmul_add_gemm(g)
    assert g.metadata["fused_matmul_add"] is True
    FusionOptimizer.fuse_gemm_relu(g)
    assert g.metadata["fused_gemm_relu"] is True
    FusionOptimizer.fuse_matmul_add_relu(g)
    assert g.metadata["fused_matmul_add_relu"] is True
    FusionOptimizer.fuse_layer_norm(g)
    assert g.metadata["fused_layer_norm"] is True
    FusionOptimizer.fuse_skip_layer_norm(g)
    assert g.metadata["fused_skip_layer_norm"] is True
    FusionOptimizer.fuse_fast_gelu(g)
    assert g.metadata["fused_fast_gelu"] is True
    FusionOptimizer.fuse_bias_gelu(g)
    assert g.metadata["fused_bias_gelu"] is True
    FusionOptimizer.fuse_mha(g)
    assert g.metadata["fused_mha"] is True
    FusionOptimizer.fuse_sdpa_mha(g)
    assert g.metadata["fused_sdpa_mha"] is True
    FusionOptimizer.fuse_rope(g)
    assert g.metadata["fused_rope"] is True
    FusionOptimizer.fuse_embed_layer_norm(g)
    assert g.metadata["fused_embed_layer_norm"] is True
    FusionOptimizer.detect_reshape_transpose(g)
    assert g.metadata["detected_reshape_transpose"] is True
    FusionOptimizer.optimize_reshape_transpose(g)
    assert g.metadata["optimized_reshape_transpose"] is True
    FusionOptimizer.cancel_identity_cast(g)
    assert g.metadata["cancelled_identity_cast"] is True
    FusionOptimizer.cancel_squeeze_unsqueeze(g)
    assert g.metadata["cancelled_squeeze_unsqueeze"] is True
    FusionOptimizer.cancel_split_concat(g)
    assert g.metadata["cancelled_split_concat"] is True
    FusionOptimizer.collapse_nested_slice(g)
    assert g.metadata["collapsed_nested_slice"] is True
    FusionOptimizer.collapse_add_constants(g)
    assert g.metadata["collapsed_add_constants"] is True
    FusionOptimizer.collapse_mul_constants(g)
    assert g.metadata["collapsed_mul_constants"] is True
    FusionOptimizer.distribute_mul_add(g)
    assert g.metadata["distributed_mul_add"] is True
    FusionOptimizer.evaluate_shape_subgraphs(g)
    assert g.metadata["evaluated_shape_subgraphs"] is True
    FusionOptimizer.deduplicate_initializers(g)
    assert g.metadata["deduplicated_initializers"] is True
    FusionOptimizer.pack_constants(g)
    assert g.metadata["packed_constants"] is True
    FusionOptimizer.generate_nhwc_conv(g)
    assert g.metadata["generated_nhwc_conv"] is True
    FusionOptimizer.generate_nhwc_maxpool(g)
    assert g.metadata["generated_nhwc_maxpool"] is True
    FusionOptimizer.convert_dropout_identity(g)
    assert g.metadata["converted_dropout_identity"] is True
    FusionOptimizer.strip_identity(g)
    assert g.metadata["stripped_identity"] is True
    FusionOptimizer.propagate_shapes(g)
    assert g.metadata["propagated_shapes"] is True
    assert FusionOptimizer.check_equivalence(g) is True
    assert "fusions" in FusionOptimizer.export_fusion_log(g)
    g2 = Graph("test2")
    g2.metadata = {}
    FusionOptimizer.run_fusions(g2, disable_rules={"fuse_conv_relu": True})
    assert "fused_conv_relu" not in g2.metadata
    FusionOptimizer.run_fusions(g2)
    assert g2.metadata["fused_conv_relu"] is True
    FusionOptimizer.custom_ops_fusions(g)
    assert g.metadata["custom_ops_fusions"] is True
