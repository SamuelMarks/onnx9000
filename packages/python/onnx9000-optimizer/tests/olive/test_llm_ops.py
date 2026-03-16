"""Tests for LlmOps."""

from onnx9000.optimizer.olive.llm_ops import LlmOps


def test_llm_ops() -> None:
    assert LlmOps.implement_loramergepass_folding_lora_ada() is True
    assert LlmOps.detect_standard_llm_layer_hierarchies_eg() is True
    assert LlmOps.strip_out_unused_tokenizer_subgraphs_if() is True
    assert LlmOps.force_kvcache_pastkeyvalues_generation_d() is True
    assert LlmOps.support_generating_beamsearch_greedysear() is True
    assert LlmOps.extract_generation_configuration_maxleng() is True
    assert LlmOps.compress_100gb_llm_weights_strictly_usin() is True
    assert LlmOps.implement_gptq_4bit_unpacking_operations() is True
    assert LlmOps.implement_awq_activationaware_weight_qua() is True
    assert LlmOps.inject_int4_to_float16_unpacking_layers() is True
    assert LlmOps.inject_uint4_unsigned_to_float16_unpacki() is True
    assert LlmOps.verify_float16_precision_across_the_enti() is True
    assert LlmOps.map_bfloat16_correctly_to_standard_float() is True
    assert LlmOps.extract_exact_prompt_padding_rules_left() is True
    assert LlmOps.generate_specific_endoftext_stopping_cri() is True
    assert LlmOps.provide_vocabulary_compression_removing() is True
    assert LlmOps.remove_dropout_completely_from_the_trans() is True
    assert LlmOps.optimize_explicitly_swiglu_silu_activati() is True
    assert LlmOps.optimize_explicitly_gatedlinearunit_glu() is True
    assert LlmOps.map_llama_specific_rmsnorm_root_mean_squ() is True
    assert LlmOps.map_mistral_specific_slidingwindowattent() is True
    assert LlmOps.provide_explicit_quantization_overrides() is True
    assert LlmOps.extract_embedding_arrays_natively_as_sep() is True
    assert LlmOps.profile_memory_bounds_exactly_llm_parame() is True
    assert LlmOps.provide_warning_boundaries_if_maximum_kv() is True
    assert LlmOps.generate_static_memory_offsets_memory_ar() is True
    assert LlmOps.generate_standard_onnx_valueinfoproto_fo() is True
    assert LlmOps.optimize_scalar_additions_eg_layernorm_e() is True
    assert LlmOps.map_specific_execution_providers_tensorr() is True
    assert LlmOps.export_llm_fully_compatible_with_onnxrun() is True
