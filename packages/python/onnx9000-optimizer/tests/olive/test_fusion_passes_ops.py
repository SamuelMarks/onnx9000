"""Tests for FusionPassesOps."""

from onnx9000.optimizer.olive.fusion_passes_ops import FusionPassesOps


def test_fusion_passes_ops() -> None:
    assert FusionPassesOps.implement_attentionfusion_standard_pytor() is True
    assert FusionPassesOps.implement_attentionfusion_with_mask_inje() is True
    assert FusionPassesOps.implement_attentionfusion_with_pastkey_p() is True
    assert FusionPassesOps.implement_attentionfusion_with_presentke() is True
    assert FusionPassesOps.implement_attentionfusion_cross_attentio() is True
    assert FusionPassesOps.implement_attentionfusion_flashattention() is True
    assert FusionPassesOps.implement_embedlayernormfusion_standard() is True
    assert FusionPassesOps.implement_embedlayernormfusion_with_word() is True
    assert FusionPassesOps.implement_skiplayernormfusion_bias_add_l() is True
    assert FusionPassesOps.implement_fastgelufusion_erf_approximati() is True
    assert FusionPassesOps.implement_fastgelufusion_tanh_approximat() is True
    assert FusionPassesOps.implement_biasgelufusion_add_gelu() is True
    assert FusionPassesOps.implement_biasdropoutfusion_add_dropout() is True
    assert FusionPassesOps.implement_nhwcconvfusion_webgpu_tensorrt() is True
    assert FusionPassesOps.implement_nhwcmaxpoolfusion() is True
    assert FusionPassesOps.implement_convaddfusion_folding_bias_nat() is True
    assert FusionPassesOps.implement_convmulfusion_folding_scale_na() is True
    assert FusionPassesOps.implement_convbatchnormfusion_mathematic() is True
    assert FusionPassesOps.implement_matmuladdfusion_creating_gemm() is True
    assert FusionPassesOps.implement_gemmrelufusion_appends_activat() is True
    assert FusionPassesOps.implement_matmuladdrelufusion_creating_g() is True
    assert FusionPassesOps.implement_reshapetransposereshapefusion() is True
    assert FusionPassesOps.implement_concatsplitfusion_canceling_ou() is True
    assert FusionPassesOps.implement_squeezeunsqueezefusion_canceli() is True
    assert FusionPassesOps.implement_padslicefusion_canceling_out_s() is True
    assert FusionPassesOps.implement_castcastfusion_canceling_redun() is True
    assert FusionPassesOps.implement_constantfolding_precalculating() is True
    assert FusionPassesOps.implement_shapeconstantfolding_precalcul() is True
    assert FusionPassesOps.implement_sliceconstantfolding() is True
    assert FusionPassesOps.implement_gatherconstantfolding() is True
    assert FusionPassesOps.implement_concatconstantfolding() is True
    assert FusionPassesOps.implement_transposeconstantfolding_bakin() is True
    assert FusionPassesOps.implement_reshapeconstantfolding_baking() is True
    assert FusionPassesOps.implement_splitconstantfolding() is True
    assert FusionPassesOps.implement_tileconstantfolding() is True
    assert FusionPassesOps.implement_expandconstantfolding() is True
    assert FusionPassesOps.detect_and_apply_rotarypositionalembeddi() is True
    assert FusionPassesOps.detect_and_apply_groupnorm_mathematical() is True
    assert FusionPassesOps.detect_and_apply_layernorm_mathematical() is True
    assert FusionPassesOps.prevent_fusions_dynamically_if_the_resul() is True
