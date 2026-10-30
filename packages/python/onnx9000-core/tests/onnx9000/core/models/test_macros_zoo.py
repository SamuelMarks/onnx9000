"""Unit tests for all architectural IR macros in `onnx9000.core.models.macros_zoo`."""

from onnx9000.core.ir import Tensor
from onnx9000.core.models.macros_zoo import (
    alphafold3_subset_module,
    alphafold_evoformer,
    anchor_free_dfl_head,
    any_resolution_vision,
    auto_correlation,
    bark_semantic_codec,
    bert_encoder_block,
    bipartite_object_query,
    bitnet_ternary_weight,
    c2f_c3_block,
    cogvideox_3d_causal_attn,
    cogvlm_visual_expert,
    controlnet_zero_conv,
    convnextv2_grn,
    dac_vq,
    deeplabv3_aspp,
    deepseek_mla,
    deepseekv3_moe_aux_loss,
    deformable_attention,
    deit_distillation_token,
    densenet_dense_block,
    edgenext_sdta,
    efficientnetv2_fused_mbconv,
    encodec_rvq,
    esm_attention,
    faster_rcnn_roi_align,
    florence2_spatial_fusion,
    flux1_rectified_flow,
    fourcastnet_afno,
    gat_layer,
    gcn_layer,
    gemma2_logit_softcapping,
    gemma_geglu,
    gemma_rmsnorm,
    gla_gated_linear_attention,
    gnn_message_passing,
    gpt2_block,
    graphcast_icosahedral,
    idefics_gated_cross_attn,
    inception_block,
    informer_probsparse_attention,
    jamba_hybrid_block,
    llama3_high_freq_rope,
    llama_gqa,
    llava_projector,
    mace_spherical_harmonics,
    mamba2_ssd,
    mamba_scan,
    maxvit_mbconv,
    minkowski_convolution,
    mobilenet_inverted_residual,
    mobilenetv4_uib,
    moe_expert_masking,
    moe_router,
    moe_topk_dispatch,
    nequip_tensor_product,
    paligemma_interleave,
    patchtst_patching,
    perceiver_resampler,
    phi_suscaled_rope,
    phi_swa,
    pna_aggregation,
    pointnet2_set_abstraction,
    pointnet_block,
    qwen1_5_dual_chunk_attention,
    qwen_qkv,
    qwenvl_2d_rope,
    repvgg_block,
    resnext_block,
    retinanet_fpn,
    retnet_retention,
    rt_detr_decoder,
    rwkv_time_mixing,
    s3d_separable_conv3d,
    sam2_memory_attention,
    sam_image_encoder,
    sam_mask_decoder,
    sam_prompt_encoder,
    sd3_mmdit,
    sd_cross_attention,
    sdxl_dual_text_encoder,
    siglip_loss,
    sparse_conv3d,
    swinv2_log_cpb,
    t5_block,
    tacotron_prenet,
    token_shift,
    unet_double_conv,
    valle_ar_nar,
    vgg_block,
    videomae_tubelet_embedding,
    vits_stochastic_duration_predictor,
    wav2vec2_feature_extractor,
    whisper_mel_spectrogram,
    xlstm_block,
    yolo_generic_block,
)


def test_yolo_and_detr_macros() -> None:
    """Verify YOLO and DETR family macro functions emit valid Tensor outputs."""
    x = Tensor("x", (1, 64, 64, 64), 1)
    w = Tensor("w", (64, 64, 3, 3), 1)

    assert isinstance(c2f_c3_block(x, w, w, w), Tensor)
    assert isinstance(repvgg_block(x, w, w), Tensor)
    assert isinstance(anchor_free_dfl_head(x, w), Tensor)
    assert isinstance(bipartite_object_query(x, x, w, w), Tensor)
    assert isinstance(deformable_attention(x, x, x), Tensor)


def test_sam_and_vlm_macros() -> None:
    """Verify SAM and VLM macro functions emit valid Tensor outputs."""
    x = Tensor("x", (1, 256, 64, 64), 1)
    shape = Tensor("shape", (4,), 7)

    assert isinstance(sam_image_encoder(x, shape, x, x, x), Tensor)
    assert isinstance(sam_prompt_encoder(x), Tensor)
    assert isinstance(sam_mask_decoder(x, x, x, x), Tensor)
    assert isinstance(perceiver_resampler(x, x, x, x), Tensor)
    assert isinstance(any_resolution_vision(x, [32, 32]), Tensor)


def test_llm_moe_and_ssm_macros() -> None:
    """Verify LLaMA, DeepSeek, Qwen, Phi, MoE, Mamba, and RWKV macros emit valid Tensor outputs."""
    x = Tensor("x", (1, 32, 128), 1)
    k = Tensor("k", (1,), 7)

    assert isinstance(llama_gqa(x, x, x, k), Tensor)
    assert isinstance(deepseek_mla(x, x, x, x), Tensor)
    assert isinstance(qwen_qkv(x, x, x, x), Tensor)
    assert isinstance(phi_suscaled_rope(x, x, x), Tensor)
    assert isinstance(phi_swa(x, x, x, x), Tensor)
    assert isinstance(moe_router(x, x), Tensor)
    assert isinstance(moe_topk_dispatch(x, k), Tensor)
    assert isinstance(moe_expert_masking(x, k, k), Tensor)
    assert isinstance(mamba_scan(x), Tensor)
    assert isinstance(rwkv_time_mixing(x, x), Tensor)
    assert isinstance(token_shift(x, k, k, k), Tensor)


def test_audio_gnn_and_foundational_macros() -> None:
    """Verify audio, GNN, BERT, GPT-2, T5, VGG, Inception, and PointNet macros."""
    x = Tensor("x", (1, 80, 300), 1)
    k = Tensor("k", (1,), 7)

    assert isinstance(whisper_mel_spectrogram(x, x, x, x), Tensor)
    assert isinstance(encodec_rvq(x, x, x), Tensor)
    assert isinstance(auto_correlation(x, k, x), Tensor)
    assert isinstance(gnn_message_passing(x, k, x), Tensor)
    assert isinstance(bert_encoder_block(x, x, x, x, x), Tensor)
    assert isinstance(gpt2_block(x, x), Tensor)
    assert isinstance(t5_block(x, x), Tensor)
    assert isinstance(vgg_block(x, x, x), Tensor)
    assert isinstance(inception_block(x, x, x, x, x), Tensor)
    assert isinstance(pointnet_block(x, x), Tensor)


def test_gemma_diffusion_and_expanded_macros() -> None:
    """Verify Gemma, MobileNet, UNet, StableDiffusion, Wav2Vec2, AlphaFold, and GNN macros."""
    x = Tensor("x", (1, 10, 128), 1)
    bind = Tensor("bind", (10,), 7)

    assert isinstance(gemma_geglu(x, x, x), Tensor)
    assert isinstance(gemma_rmsnorm(x, x), Tensor)
    assert isinstance(mamba2_ssd(x), Tensor)
    assert isinstance(mobilenet_inverted_residual(x, x, x, x), Tensor)
    assert isinstance(unet_double_conv(x, x, x), Tensor)
    assert isinstance(sd_cross_attention(x, x, x, x, x), Tensor)
    assert isinstance(wav2vec2_feature_extractor(x, x, x), Tensor)
    assert isinstance(gemma2_logit_softcapping(x, 10.0), Tensor)
    assert isinstance(bitnet_ternary_weight(x), Tensor)
    assert isinstance(convnextv2_grn(x, x, x), Tensor)
    assert isinstance(faster_rcnn_roi_align(x, x, bind), Tensor)
    assert isinstance(gcn_layer(x, x, x), Tensor)
    assert isinstance(gat_layer(x, x, x, x), Tensor)
    assert isinstance(alphafold_evoformer(x, x, x, x), Tensor)


def test_nextgen_and_experimental_macros() -> None:
    """Verify Qwen1.5, LLaMA3, DeepSeekV3, Jamba, DenseNet, DeepLab, SwinV2, SD3, Flux, and xLSTM macros."""
    x = Tensor("x", (1, 10, 128), 1)
    ind = Tensor("ind", (1, 5, 128), 7)

    assert isinstance(qwen1_5_dual_chunk_attention(x, x, x, x), Tensor)
    assert isinstance(llama3_high_freq_rope(x, x, 2.0), Tensor)
    assert isinstance(deepseekv3_moe_aux_loss(x, x), Tensor)
    assert isinstance(jamba_hybrid_block(x, True), Tensor)
    assert isinstance(jamba_hybrid_block(x, False), Tensor)
    assert isinstance(densenet_dense_block(x, x), Tensor)
    assert isinstance(deeplabv3_aspp(x, x, x), Tensor)
    assert isinstance(swinv2_log_cpb(x, x), Tensor)
    assert isinstance(sd3_mmdit(x, x, x), Tensor)
    assert isinstance(flux1_rectified_flow(x, x, x), Tensor)
    assert isinstance(pointnet2_set_abstraction(x, x, x), Tensor)
    assert isinstance(xlstm_block(x, x, x, x, x), Tensor)
    assert isinstance(retnet_retention(x, x, x, x), Tensor)
    assert isinstance(gla_gated_linear_attention(x, x, x, x), Tensor)
    assert isinstance(s3d_separable_conv3d(x, x, x), Tensor)
    assert isinstance(videomae_tubelet_embedding(x, x), Tensor)
    assert isinstance(informer_probsparse_attention(x, x, x, ind), Tensor)
    assert isinstance(fourcastnet_afno(x, x, x), Tensor)
    assert isinstance(vits_stochastic_duration_predictor(x, x, x), Tensor)


def test_remaining_and_final_macros() -> None:
    """Verify DeiT, MaxViT, EdgeNeXt, RT-DETR, SAM2, Florence2, LLaVA, PaliGemma, Idefics, and scientific macros."""
    x = Tensor("x", (1, 10, 128), 1)

    assert isinstance(deit_distillation_token(x, x, x), Tensor)
    assert isinstance(maxvit_mbconv(x, x), Tensor)
    assert isinstance(edgenext_sdta(x, x), Tensor)
    assert isinstance(yolo_generic_block(x, x), Tensor)
    assert isinstance(retinanet_fpn(x, x, x), Tensor)
    assert isinstance(rt_detr_decoder(x, x), Tensor)
    assert isinstance(sam2_memory_attention(x, x, x, x), Tensor)
    assert isinstance(florence2_spatial_fusion(x, x), Tensor)
    assert isinstance(llava_projector(x, x, x), Tensor)
    assert isinstance(paligemma_interleave(x, x), Tensor)
    assert isinstance(idefics_gated_cross_attn(x, x, x, x), Tensor)
    assert isinstance(tacotron_prenet(x, x), Tensor)
    assert isinstance(minkowski_convolution(x, x, x), Tensor)
    assert isinstance(pna_aggregation(x, x), Tensor)
    assert isinstance(nequip_tensor_product(x, x), Tensor)
    assert isinstance(alphafold3_subset_module(x, x), Tensor)
    assert isinstance(resnext_block(x, x), Tensor)
    assert isinstance(mobilenetv4_uib(x, x, x), Tensor)
    assert isinstance(efficientnetv2_fused_mbconv(x, x), Tensor)
    assert isinstance(siglip_loss(x, x), Tensor)
    assert isinstance(qwenvl_2d_rope(x, x, x), Tensor)
    assert isinstance(cogvlm_visual_expert(x, x), Tensor)
    assert isinstance(sdxl_dual_text_encoder(x, x), Tensor)
    assert isinstance(controlnet_zero_conv(x, x), Tensor)
    assert isinstance(cogvideox_3d_causal_attn(x, x, x, x), Tensor)
    assert isinstance(bark_semantic_codec(x, x), Tensor)
    assert isinstance(valle_ar_nar(x, x), Tensor)
    assert isinstance(dac_vq(x, x), Tensor)
    assert isinstance(sparse_conv3d(x, x), Tensor)
    assert isinstance(mace_spherical_harmonics(x, x), Tensor)
    assert isinstance(patchtst_patching(x, 16), Tensor)
    assert isinstance(graphcast_icosahedral(x, x), Tensor)
    assert isinstance(esm_attention(x, x, x), Tensor)
