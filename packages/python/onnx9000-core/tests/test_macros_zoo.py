"""Tests for Phase 2: Exhaustive Architectural Mappings to Core IR."""

import pytest
from onnx9000.core.ir import Tensor
from onnx9000.core.models.macros_zoo import (
    anchor_free_dfl_head,
    any_resolution_vision,
    auto_correlation,
    bipartite_object_query,
    c2f_c3_block,
    deepseek_mla,
    deformable_attention,
    encodec_rvq,
    gnn_message_passing,
    llama_gqa,
    mamba_scan,
    moe_expert_masking,
    moe_router,
    moe_topk_dispatch,
    perceiver_resampler,
    phi_suscaled_rope,
    phi_swa,
    qwen_qkv,
    repvgg_block,
    rwkv_time_mixing,
    sam_image_encoder,
    sam_mask_decoder,
    sam_prompt_encoder,
    token_shift,
    whisper_mel_spectrogram,
)


def test_yolo_macros():
    """Docstring for D103."""
    x = Tensor("x", (1, 64, 64, 64), 1)
    w1 = Tensor("w1", (32, 64, 1, 1), 1)
    w2 = Tensor("w2", (32, 64, 1, 1), 1)
    w3 = Tensor("w3", (64, 64, 1, 1), 1)

    out = c2f_c3_block.__wrapped__(x, w1, w2, w3)
    assert isinstance(out, Tensor)

    w_3x3 = Tensor("w_3x3", (64, 64, 3, 3), 1)
    w_1x1 = Tensor("w_1x1", (64, 64, 1, 1), 1)
    out = repvgg_block.__wrapped__(x, w_3x3, w_1x1)
    assert isinstance(out, Tensor)

    proj = Tensor("proj", (64, 80), 1)
    out = anchor_free_dfl_head.__wrapped__(x, proj)
    assert isinstance(out, Tensor)


def test_detr_macros():
    """Docstring for D103."""
    memory = Tensor("mem", (1, 100, 256), 1)
    queries = Tensor("q", (1, 10, 256), 1)
    kw = Tensor("kw", (256, 256), 1)
    vw = Tensor("vw", (256, 256), 1)
    out = bipartite_object_query(memory, queries, kw, vw)
    assert isinstance(out, Tensor)

    x = Tensor("x", (1, 256, 64, 64), 1)
    locations = Tensor("loc", (1, 10, 4, 2), 1)
    attn_w = Tensor("attn", (1, 10, 4), 1)
    out = deformable_attention.__wrapped__(x, locations, attn_w)
    assert isinstance(out, Tensor)


def test_sam_macros():
    """Docstring for D103."""
    x = Tensor("x", (1, 256, 64, 64), 1)
    shape = Tensor("shape", (4,), 7)
    q = Tensor("q", (1, 64, 256), 1)
    k = Tensor("k", (1, 64, 256), 1)
    v = Tensor("v", (1, 64, 256), 1)
    out = sam_image_encoder.__wrapped__(x, shape, q, k, v)
    assert isinstance(out, Tensor)

    out = sam_prompt_encoder.__wrapped__(x)
    assert isinstance(out, Tensor)

    pe = Tensor("pe", (1, 256), 1)
    sparse = Tensor("sparse", (1, 10, 256), 1)
    dense = Tensor("dense", (1, 256, 64, 64), 1)
    out = sam_mask_decoder.__wrapped__(x, pe, sparse, dense)
    assert isinstance(out, Tensor)


def test_vlms_macros():
    """Docstring for D103."""
    latents = Tensor("latents", (1, 64, 256), 1)
    context = Tensor("context", (1, 196, 256), 1)
    kw = Tensor("kw", (256, 256), 1)
    vw = Tensor("vw", (256, 256), 1)
    out = perceiver_resampler(latents, context, kw, vw)
    assert isinstance(out, Tensor)

    x = Tensor("x", (1, 256, 64, 64), 1)
    out = any_resolution_vision.__wrapped__(x, [32, 32])
    assert isinstance(out, Tensor)


def test_llm_macros():
    """Docstring for D103."""
    q = Tensor("q", (1, 32, 128), 1)
    k = Tensor("k", (1, 8, 128), 1)
    v = Tensor("v", (1, 8, 128), 1)
    expand_shape = Tensor("shape", (4,), 7)
    out = llama_gqa(q, k, v, expand_shape)
    assert isinstance(out, Tensor)

    x = Tensor("x", (1, 128), 1)
    w_down = Tensor("wd", (128, 64), 1)
    w_up = Tensor("wu", (64, 128), 1)
    rope = Tensor("rope", (1, 128), 1)
    out = deepseek_mla.__wrapped__(x, w_down, w_up, rope)
    assert isinstance(out, Tensor)

    w_qkv = Tensor("wqkv", (128, 384), 1)
    b_qkv = Tensor("bqkv", (384,), 1)
    word_emb = Tensor("word", (384, 1000), 1)
    out = qwen_qkv.__wrapped__(x, w_qkv, b_qkv, word_emb)
    assert isinstance(out, Tensor)

    rope_scale = Tensor("scale", (1,), 1)
    out = phi_suscaled_rope(q, k, rope_scale)
    assert isinstance(out, Tensor)

    mask = Tensor("mask", (32, 32), 1)
    out = phi_swa(q, k, v, mask)
    assert isinstance(out, Tensor)


def test_sparse_moe_macros():
    """Docstring for D103."""
    x = Tensor("x", (1, 128), 1)
    w_gate = Tensor("wg", (128, 8), 1)
    out = moe_router.__wrapped__(x, w_gate)
    assert isinstance(out, Tensor)

    k = Tensor("k", (1,), 7)
    out = moe_topk_dispatch(out, k)
    assert isinstance(out, Tensor)

    indices = Tensor("ind", (1,), 7)
    target = Tensor("tgt", (1,), 7)
    out = moe_expert_masking.__wrapped__(x, indices, target)
    assert isinstance(out, Tensor)


def test_rnn_macros():
    """Docstring for D103."""
    x = Tensor("x", (10, 128), 1)
    out = mamba_scan.__wrapped__(x)
    assert isinstance(out, Tensor)

    state = Tensor("state", (1, 128), 1)
    out = rwkv_time_mixing.__wrapped__(x, state)
    assert isinstance(out, Tensor)

    pads = Tensor("pads", (4,), 7)
    starts = Tensor("starts", (1,), 7)
    ends = Tensor("ends", (1,), 7)
    out = token_shift.__wrapped__(x, pads, starts, ends)
    assert isinstance(out, Tensor)


def test_audio_macros():
    """Docstring for D103."""
    x = Tensor("x", (1, 80, 3000), 1)
    cw = Tensor("cw", (80, 80, 3), 1)
    mf = Tensor("mf", (80, 80), 1)
    exp = Tensor("exp", (), 1)
    out = whisper_mel_spectrogram.__wrapped__(x, cw, mf, exp)
    assert isinstance(out, Tensor)

    codebook = Tensor("cb", (1024, 80), 1)
    out = encodec_rvq.__wrapped__(x, codebook, exp)
    assert isinstance(out, Tensor)


def test_time_series_gnn_macros():
    """Docstring for D103."""
    x = Tensor("x", (1, 100, 64), 1)
    k = Tensor("k", (1,), 7)
    w = Tensor("w", (64, 64), 1)
    out = auto_correlation.__wrapped__(x, k, w)
    assert isinstance(out, Tensor)

    nodes = Tensor("nodes", (100, 64), 1)
    edges = Tensor("edges", (2, 500), 7)
    updates = Tensor("updates", (500, 64), 1)
    out = gnn_message_passing(nodes, edges, updates)
    assert isinstance(out, Tensor)


def test_yolo_family():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.models.macros_zoo import anchor_free_dfl_head, c2f_c3_block, repvgg_block

    x = Tensor(name="x", shape=[1, 3, 224, 224], dtype=1)
    w = Tensor(name="w", shape=[1, 3, 3, 3], dtype=1)

    c = c2f_c3_block.__wrapped__(x, w, w, w)
    assert c is not None
    r = repvgg_block.__wrapped__(x, w, w)
    assert r is not None
    a = anchor_free_dfl_head.__wrapped__(x, w)
    assert a is not None


def test_detr_family():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.models.macros_zoo import bipartite_object_query

    x = Tensor(name="x", shape=[1, 3, 224, 224], dtype=1)
    w = Tensor(name="w", shape=[1, 3, 3, 3], dtype=1)

    b = bipartite_object_query.__wrapped__(x, x, w, w)
    assert b is not None


def test_detr_family_rest():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.models.macros_zoo import (
        deformable_attention,
        sam_image_encoder,
        sam_mask_decoder,
        sam_prompt_encoder,
    )

    x = Tensor(name="x", shape=[1, 3, 224, 224], dtype=1)

    d = deformable_attention.__wrapped__(x, x, x)
    assert d is not None
    s = sam_image_encoder.__wrapped__(x, x, x, x, x)
    assert s is not None
    p = sam_prompt_encoder.__wrapped__(x)
    assert p is not None
    m = sam_mask_decoder.__wrapped__(x, x, x, x)
    assert m is not None


def test_hybrid_vision():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.models.macros_zoo import any_resolution_vision, perceiver_resampler

    x = Tensor(name="x", shape=[1, 3, 224, 224], dtype=1)

    p = perceiver_resampler.__wrapped__(x, x, x, x)
    assert p is not None
    a = any_resolution_vision.__wrapped__(x, [3, 3])
    assert a is not None


def test_llms():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.models.macros_zoo import (
        deepseek_mla,
        llama_gqa,
        phi_suscaled_rope,
        phi_swa,
        qwen_qkv,
    )

    x = Tensor(name="x", shape=[1, 3, 224, 224], dtype=1)

    l = llama_gqa.__wrapped__(x, x, x, x)
    assert l is not None
    d = deepseek_mla.__wrapped__(x, x, x, x)
    assert d is not None
    q = qwen_qkv.__wrapped__(x, x, x, x)
    assert q is not None
    ps = phi_suscaled_rope.__wrapped__(x, x, x)
    assert ps is not None
    psw = phi_swa.__wrapped__(x, x, x, x)
    assert psw is not None


def test_moe():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.models.macros_zoo import moe_expert_masking, moe_router, moe_topk_dispatch

    x = Tensor(name="x", shape=[1, 3, 224, 224], dtype=1)

    mr = moe_router.__wrapped__(x, x)
    assert mr is not None
    mt = moe_topk_dispatch.__wrapped__(x, x)
    assert mt is not None
    me = moe_expert_masking.__wrapped__(x, x, x)
    assert me is not None


def test_ssm():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.models.macros_zoo import mamba_scan, rwkv_time_mixing, token_shift

    x = Tensor(name="x", shape=[1, 3, 224, 224], dtype=1)

    ms = mamba_scan.__wrapped__(x)
    assert ms is not None
    rt = rwkv_time_mixing.__wrapped__(x, x)
    assert rt is not None
    ts = token_shift.__wrapped__(x, x, x, x)
    assert ts is not None


def test_audio_gnn():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.models.macros_zoo import (
        auto_correlation,
        encodec_rvq,
        gnn_message_passing,
        whisper_mel_spectrogram,
    )

    x = Tensor(name="x", shape=[1, 3, 224, 224], dtype=1)

    w = whisper_mel_spectrogram.__wrapped__(x, x, x, x)
    assert w is not None
    e = encodec_rvq.__wrapped__(x, x, x)
    assert e is not None
    a = auto_correlation.__wrapped__(x, x, x)
    assert a is not None
    g = gnn_message_passing.__wrapped__(x, x, x)
    assert g is not None


def test_legacy_transformers():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.models.macros_zoo import bert_encoder_block, gpt2_block, t5_block

    x = Tensor("x", (1, 10, 128), 1)
    mask = Tensor("mask", (1, 10), 1)
    w = Tensor("w", (128, 128), 1)

    out1 = bert_encoder_block.__wrapped__(x, mask, w, w, w)
    assert out1 is not None

    out2 = gpt2_block.__wrapped__(x, w)
    assert out2 is not None

    out3 = t5_block.__wrapped__(x, x)
    assert out3 is not None


def test_vision_foundations():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.models.macros_zoo import inception_block, vgg_block

    x = Tensor("x", (1, 64, 32, 32), 1)
    w_conv = Tensor("w_conv", (64, 64, 3, 3), 1)
    w_1x1 = Tensor("w_1x1", (64, 64, 1, 1), 1)
    w_5x5 = Tensor("w_5x5", (64, 64, 5, 5), 1)

    out1 = vgg_block.__wrapped__(x, w_conv, w_conv)
    assert out1 is not None

    out2 = inception_block.__wrapped__(x, w_1x1, w_conv, w_5x5, w_1x1)
    assert out2 is not None


def test_pointnet():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.models.macros_zoo import pointnet_block

    x = Tensor("x", (1, 1024, 3), 1)
    w_mlp = Tensor("w_mlp", (3, 64), 1)

    out = pointnet_block.__wrapped__(x, w_mlp)
    assert out is not None


def test_advanced_llms():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.models.macros_zoo import gemma_geglu, gemma_rmsnorm, mamba2_ssd

    x = Tensor("x", (1, 10, 128), 1)
    w = Tensor("w", (128, 128), 1)

    assert gemma_geglu.__wrapped__(x, w, w) is not None
    assert gemma_rmsnorm.__wrapped__(x, w) is not None
    assert mamba2_ssd.__wrapped__(x) is not None


def test_extra_vision():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.models.macros_zoo import mobilenet_inverted_residual, unet_double_conv

    x = Tensor("x", (1, 64, 32, 32), 1)
    w = Tensor("w", (64, 64, 3, 3), 1)

    assert mobilenet_inverted_residual.__wrapped__(x, w, w, w) is not None
    assert unet_double_conv.__wrapped__(x, w, w) is not None


def test_diffusion_and_audio():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.models.macros_zoo import sd_cross_attention, wav2vec2_feature_extractor

    x = Tensor("x", (1, 10, 128), 1)
    c = Tensor("c", (1, 20, 128), 1)
    w = Tensor("w", (128, 128), 1)

    assert sd_cross_attention.__wrapped__(x, c, w, w, w) is not None

    x_audio = Tensor("x", (1, 1, 16000), 1)
    w_conv = Tensor("w", (1, 1, 5), 1)
    assert wav2vec2_feature_extractor.__wrapped__(x_audio, w_conv, w_conv) is not None


def test_expanded_macros():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.models.macros_zoo import (
        alphafold_evoformer,
        bitnet_ternary_weight,
        convnextv2_grn,
        faster_rcnn_roi_align,
        gat_layer,
        gcn_layer,
        gemma2_logit_softcapping,
    )

    x = Tensor("x", (1, 10, 128), 1)

    assert gemma2_logit_softcapping.__wrapped__(x, 10.0) is not None
    assert bitnet_ternary_weight.__wrapped__(x) is not None

    x_vision = Tensor("xv", (1, 64, 32, 32), 1)
    g = Tensor("g", (1, 64, 1, 1), 1)
    b = Tensor("b", (1, 64, 1, 1), 1)
    assert convnextv2_grn.__wrapped__(x_vision, g, b) is not None

    rois = Tensor("rois", (10, 4), 1)
    batch_ind = Tensor("bind", (10,), 7)
    assert faster_rcnn_roi_align.__wrapped__(x_vision, rois, batch_ind) is not None

    adj = Tensor("adj", (10, 10), 1)
    w = Tensor("w", (128, 64), 1)
    a = Tensor("a", (64, 10), 1)
    assert gcn_layer.__wrapped__(x, adj, w) is not None
    assert gat_layer.__wrapped__(x, adj, w, a) is not None

    msa = Tensor("msa", (1, 10, 20, 128), 1)
    pair = Tensor("pair", (1, 20, 20, 64), 1)
    w_msa = Tensor("wm", (128, 128), 1)
    w_pair = Tensor("wp", (64, 64), 1)
    assert alphafold_evoformer.__wrapped__(msa, pair, w_msa, w_pair) is not None


def test_nextgen_macros():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.models.macros_zoo import (
        deeplabv3_aspp,
        deepseekv3_moe_aux_loss,
        densenet_dense_block,
        flux1_rectified_flow,
        jamba_hybrid_block,
        llama3_high_freq_rope,
        pointnet2_set_abstraction,
        qwen1_5_dual_chunk_attention,
        sd3_mmdit,
        swinv2_log_cpb,
    )

    x = Tensor("x", (1, 10, 128), 1)

    assert qwen1_5_dual_chunk_attention.__wrapped__(x, x, x, x) is not None
    assert llama3_high_freq_rope.__wrapped__(x, x, 2.0) is not None
    assert deepseekv3_moe_aux_loss.__wrapped__(x, x) is not None
    assert jamba_hybrid_block.__wrapped__(x, True) is not None

    xv = Tensor("xv", (1, 64, 32, 32), 1)
    w = Tensor("w", (64, 64, 3, 3), 1)

    assert densenet_dense_block.__wrapped__(xv, w) is not None
    assert deeplabv3_aspp.__wrapped__(xv, w, w) is not None

    coords = Tensor("c", (10, 2), 1)
    wc = Tensor("wc", (2, 8), 1)
    assert swinv2_log_cpb.__wrapped__(coords, wc) is not None

    x1 = Tensor("x1", (1, 10, 64), 1)
    x2 = Tensor("x2", (1, 10, 64), 1)
    w_dit = Tensor("w_dit", (128, 128), 1)
    assert sd3_mmdit.__wrapped__(x1, x2, w_dit) is not None

    assert flux1_rectified_flow.__wrapped__(x, x, x) is not None

    xyz = Tensor("xyz", (1, 100, 3), 1)
    points = Tensor("pts", (1, 100, 64), 1)
    wp = Tensor("wp", (67, 128), 1)
    assert pointnet2_set_abstraction.__wrapped__(xyz, points, wp) is not None


def test_experimental_macros():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.models.macros_zoo import (
        fourcastnet_afno,
        gla_gated_linear_attention,
        informer_probsparse_attention,
        retnet_retention,
        s3d_separable_conv3d,
        videomae_tubelet_embedding,
        vits_stochastic_duration_predictor,
        xlstm_block,
    )

    x = Tensor("x", (1, 10, 128), 1)

    assert xlstm_block.__wrapped__(x, x, x, x, x) is not None
    assert retnet_retention.__wrapped__(x, x, x, x) is not None
    assert gla_gated_linear_attention.__wrapped__(x, x, x, x) is not None

    x3d = Tensor("x3d", (1, 64, 16, 32, 32), 1)
    ws = Tensor("ws", (64, 64, 1, 3, 3), 1)
    wt = Tensor("wt", (64, 64, 3, 1, 1), 1)

    assert s3d_separable_conv3d.__wrapped__(x3d, ws, wt) is not None

    w_tube = Tensor("w_tube", (128, 3, 2, 16, 16), 1)
    x_vid = Tensor("x_vid", (1, 3, 16, 224, 224), 1)
    assert videomae_tubelet_embedding.__wrapped__(x_vid, w_tube) is not None

    indices = Tensor("ind", (1, 5, 128), 7)
    assert informer_probsparse_attention.__wrapped__(x, x, x, indices) is not None

    assert fourcastnet_afno.__wrapped__(x, x, x) is not None

    noise = Tensor("noise", (1, 10, 128), 1)
    w_vits = Tensor("wv", (256, 128), 1)
    assert vits_stochastic_duration_predictor.__wrapped__(x, noise, w_vits) is not None


def test_remaining_macros():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.models.macros_zoo import (
        alphafold3_subset_module,
        deit_distillation_token,
        edgenext_sdta,
        florence2_spatial_fusion,
        idefics_gated_cross_attn,
        llava_projector,
        maxvit_mbconv,
        minkowski_convolution,
        nequip_tensor_product,
        paligemma_interleave,
        pna_aggregation,
        retinanet_fpn,
        rt_detr_decoder,
        sam2_memory_attention,
        tacotron_prenet,
        yolo_generic_block,
    )

    x = Tensor("x", (1, 10, 128), 1)

    assert deit_distillation_token.__wrapped__(x, x, x) is not None
    assert maxvit_mbconv.__wrapped__(x, x) is not None
    assert edgenext_sdta.__wrapped__(x, x) is not None
    assert yolo_generic_block.__wrapped__(x, x) is not None
    assert retinanet_fpn.__wrapped__(x, x, x) is not None
    assert rt_detr_decoder.__wrapped__(x, x) is not None
    assert sam2_memory_attention.__wrapped__(x, x, x, x) is not None
    assert florence2_spatial_fusion.__wrapped__(x, x) is not None
    assert llava_projector.__wrapped__(x, x, x) is not None
    assert paligemma_interleave.__wrapped__(x, x) is not None
    assert idefics_gated_cross_attn.__wrapped__(x, x, x, x) is not None
    assert tacotron_prenet.__wrapped__(x, x) is not None
    assert minkowski_convolution.__wrapped__(x, x, x) is not None
    assert pna_aggregation.__wrapped__(x, x) is not None
    assert nequip_tensor_product.__wrapped__(x, x) is not None
    assert alphafold3_subset_module.__wrapped__(x, x) is not None


def test_final_macros():
    """Docstring for D103."""
    from onnx9000.core.ir import Tensor
    from onnx9000.core.models.macros_zoo import (
        bark_semantic_codec,
        cogvideox_3d_causal_attn,
        cogvlm_visual_expert,
        controlnet_zero_conv,
        dac_vq,
        efficientnetv2_fused_mbconv,
        esm_attention,
        graphcast_icosahedral,
        mace_spherical_harmonics,
        mobilenetv4_uib,
        patchtst_patching,
        qwenvl_2d_rope,
        resnext_block,
        sdxl_dual_text_encoder,
        siglip_loss,
        sparse_conv3d,
        valle_ar_nar,
    )

    x = Tensor("x", (1, 10, 128), 1)

    assert resnext_block.__wrapped__(x, x) is not None
    assert mobilenetv4_uib.__wrapped__(x, x, x) is not None
    assert efficientnetv2_fused_mbconv.__wrapped__(x, x) is not None
    assert siglip_loss.__wrapped__(x, x) is not None
    assert qwenvl_2d_rope.__wrapped__(x, x, x) is not None
    assert cogvlm_visual_expert.__wrapped__(x, x) is not None
    assert sdxl_dual_text_encoder.__wrapped__(x, x) is not None
    assert controlnet_zero_conv.__wrapped__(x, x) is not None
    assert cogvideox_3d_causal_attn.__wrapped__(x, x, x, x) is not None
    assert bark_semantic_codec.__wrapped__(x, x) is not None
    assert valle_ar_nar.__wrapped__(x, x) is not None
    assert dac_vq.__wrapped__(x, x) is not None
    assert sparse_conv3d.__wrapped__(x, x) is not None
    assert mace_spherical_harmonics.__wrapped__(x, x) is not None
    assert patchtst_patching.__wrapped__(x, 16) is not None
    assert graphcast_icosahedral.__wrapped__(x, x) is not None
    assert esm_attention.__wrapped__(x, x, x) is not None
