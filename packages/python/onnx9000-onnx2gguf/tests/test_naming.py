"""Tests for naming."""

import pytest
from onnx9000.onnx2gguf.naming import rename_tensor


def test_rename_tensor():
    """Provides functional implementation."""
    assert rename_tensor("model.embed_tokens.weight") == "token_embd.weight"
    assert rename_tensor("model.layers.0.input_layernorm.weight") == "blk.0.attn_norm.weight"
    assert rename_tensor("model.layers.10.self_attn.q_proj.weight") == "blk.10.attn_q.weight"
    assert rename_tensor("model.layers.5.self_attn.k_proj.bias") == "blk.5.attn_k.bias"
    assert rename_tensor("model.layers.2.self_attn.v_proj.weight") == "blk.2.attn_v.weight"
    assert rename_tensor("model.layers.1.self_attn.o_proj.weight") == "blk.1.attn_output.weight"
    assert rename_tensor("model.layers.3.self_attn.qkv_proj.weight") == "blk.3.attn_qkv.weight"
    assert (
        rename_tensor("model.layers.4.post_attention_layernorm.weight") == "blk.4.ffn_norm.weight"
    )
    assert rename_tensor("model.layers.6.mlp.gate_proj.weight") == "blk.6.ffn_gate.weight"
    assert rename_tensor("model.layers.7.mlp.down_proj.weight") == "blk.7.ffn_down.weight"
    assert rename_tensor("model.layers.8.mlp.up_proj.weight") == "blk.8.ffn_up.weight"
    assert rename_tensor("model.layers.9.mlp.gate_up_proj.weight") == "blk.9.ffn_gate_up.weight"
    assert rename_tensor("model.layers.1.ffn_gate_inp.weight") == "blk.1.ffn_gate_inp.weight"
    assert rename_tensor("model.norm.weight") == "output_norm.weight"
    assert rename_tensor("lm_head.weight") == "output.weight"

    # Overrides
    assert rename_tensor("custom.weight", {r"^custom\.weight$": "custom_mapped"}) == "custom_mapped"

    with pytest.raises(ValueError):
        rename_tensor("unknown.tensor")
